import argparse
from datetime import datetime
import shutil

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.rich import tqdm
import wandb

import src.config as config
from src.models import get_model, get_data_mode
from src.training.dataset import MahjongDataset
import src.utils as utils


class MahjongTrainer:
    """
    Trainer for Mahjong AI.
    """

    def __init__(self, arch: str, device: str, run_name: str):
        """
        Initialize trainer components.
        """
        # Set random seed
        utils.set_seed(config.SEED)

        self.arch = arch
        self.device = utils.get_device(device)
        self.run_name = run_name

        # Set saving directory
        self.save_dir = config.MODEL_DIR / self.arch / self.run_name
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Copy config file
        shutil.copy(config.SRC_ROOT / "config.py", self.save_dir / "config.py")

        # Determine data representation
        self.mode = get_data_mode(arch)

        # Initialize datasets
        self.train_dataset = MahjongDataset(mode=self.mode, split="train")
        self.valid_dataset = MahjongDataset(mode=self.mode, split="valid")

        # Initialize dataloader
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=True,
            num_workers=config.NUM_PROCESSES,
            pin_memory=True,
            persistent_workers=True,
        )
        self.valid_loader = DataLoader(
            self.valid_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=config.NUM_PROCESSES,
            pin_memory=True,
            persistent_workers=True,
        )

        # Build model
        self.model = get_model(self.arch).to(self.device)

        # Set up optimizer, loss function, and scaler
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.LEARNING_RATE)
        self.criterion_ce = nn.CrossEntropyLoss()
        self.criterion_bce = nn.BCEWithLogitsLoss(reduction="none")
        self.scaler = torch.amp.GradScaler("cuda") if self.device.type == "cuda" else None

        # Initialize WandB
        wandb.init(
            project=config.WANDB_PROJECT,
            entity=config.WANDB_ENTITY,
            group=self.arch,
            job_type="train",
            name=self.run_name,
            config={
                "architecture": self.arch,
                "num_parameters": sum(p.numel() for p in self.model.parameters()),
                "batch_size": config.BATCH_SIZE,
                "learning_rate": config.LEARNING_RATE,
                "max_epochs": config.MAX_EPOCHS,
                "early_stopping_patience": config.EARLY_STOPPING_PATIENCE,
                "lambda_action": config.LAMBDA_ACTION,
                "lambda_intent": config.LAMBDA_INTENT,
                "lambda_belief": config.LAMBDA_BELIEF,
            },
        )

    def compute_loss(self, output: dict, batch: dict) -> torch.Tensor:
        """
        Compute combined loss for primary and auxiliary tasks.
        """
        action_list = batch["action"].to(self.device, non_blocking=True)

        # Primary task: Action prediction
        loss = config.LAMBDA_ACTION * self.criterion_ce(output["action"], action_list)

        # Auxiliary task 1: Intent prediction (Yaku prediction)
        if "intent" in output:
            yaku = batch["yaku"].to(self.device, non_blocking=True).float()
            yaku_mask = batch["yaku_mask"].to(self.device, non_blocking=True).float()
            raw_loss_intent = self.criterion_bce(output["intent"], yaku)
            masked_loss_intent = (raw_loss_intent.mean(dim=1) * yaku_mask).sum() / (yaku_mask.sum() + 1e-8)
            loss += config.LAMBDA_INTENT * masked_loss_intent

        # Auxiliary task 2: Belief prediction (Opponent shanten number prediction)
        if "belief" in output:
            opponent_shanten_numbers = batch["opponent_shanten_numbers"].to(self.device, non_blocking=True).long()
            logits_belief = output["belief"].view(-1, config.NUM_SHANTEN_CLASSES)
            targets_belief = opponent_shanten_numbers.view(-1)
            loss += config.LAMBDA_BELIEF * self.criterion_ce(logits_belief, targets_belief)

        return loss

    def train_epoch(self, epoch: int) -> tuple[float, float]:
        """
        Perform one training epoch.
        """
        # Set model to training mode
        self.model.train()
        total_loss: float = 0.0
        total_correct: int = 0
        total_sample: int = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]", leave=False)

        for batch in pbar:
            # Move data to target device
            states = batch["state"].to(self.device, non_blocking=True)
            actions = batch["action"].to(self.device, non_blocking=True)

            # Clear gradients
            self.optimizer.zero_grad(set_to_none=True)

            # Forward pass with mixed precision
            if self.scaler:
                with torch.amp.autocast("cuda"):
                    output = self.model(states)
                    loss = self.compute_loss(output, batch)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

            else:
                output = self.model(states)
                loss = self.compute_loss(output, batch)
                loss.backward()
                self.optimizer.step()

            # Record metrics
            total_loss += loss.item()
            _, predicted = output["action"].max(1)
            total_sample += actions.size(0)
            total_correct += int(predicted.eq(actions).sum().item())

        return total_loss / len(self.train_loader), total_correct / total_sample

    @torch.no_grad()
    def valid_epoch(self, epoch: int) -> tuple[float, float]:
        """
        Perform one validation epoch.
        """
        # Set model to evaluation mode
        self.model.eval()
        total_loss: float = 0.0
        total_correct: int = 0
        total_sample: int = 0

        pbar = tqdm(self.valid_loader, desc=f"Epoch {epoch} [Valid]", leave=False)

        for batch in pbar:
            # Move data to device
            states = batch["state"].to(self.device, non_blocking=True)
            actions = batch["action"].to(self.device, non_blocking=True)

            # Inference with mixed precision
            if self.scaler:
                with torch.amp.autocast("cuda"):
                    output = self.model(states)
                    loss = self.compute_loss(output, batch)

            else:
                output = self.model(states)
                loss = self.compute_loss(output, batch)

            # Record metrics
            total_loss += loss.item()
            _, predicted = output["action"].max(1)
            total_sample += actions.size(0)
            total_correct += int(predicted.eq(actions).sum().item())

        return total_loss / len(self.valid_loader), total_correct / total_sample

    def save_checkpoint(self, filename: str):
        """
        Save the model checkpoint.
        """
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "arch": self.arch,
            },
            self.save_dir / filename,
        )

    def run(self):
        """
        Execute the full training process.
        """
        best_accuracy = 0.0
        best_epoch = 0
        patience_counter = 0

        # Main training loop
        epoch_pbar = tqdm(range(1, config.MAX_EPOCHS + 1), desc="Training")

        for epoch in epoch_pbar:
            # Training and Validation steps
            train_loss, train_accuracy = self.train_epoch(epoch)
            valid_loss, valid_accuracy = self.valid_epoch(epoch)

            # Update best accuracy
            if valid_accuracy > best_accuracy:
                best_accuracy = valid_accuracy
                best_epoch = epoch
                self.save_checkpoint("best_model.pth")
                patience_counter = 0

            else:
                patience_counter += 1

            # Log to WandB
            wandb.log(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_accuracy": train_accuracy,
                    "valid_loss": valid_loss,
                    "valid_accuracy": valid_accuracy,
                    "best_epoch": best_epoch,
                    "best_accuracy": best_accuracy,
                    "patience": patience_counter,
                }
            )

            # Update progress bar
            epoch_pbar.set_description(
                f"Training (Best: {best_accuracy:.4f} @ Epoch {best_epoch}, Patience: {patience_counter}/{config.EARLY_STOPPING_PATIENCE})"
            )

            # Early stopping check
            if patience_counter >= config.EARLY_STOPPING_PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break

        # Finish tracking
        wandb.finish()


if __name__ == "__main__":
    # Set up logging
    utils.setup_logging()

    # Parse argsuments
    parser = argparse.ArgumentParser(description="Train Mahjong AI.")
    parser.add_argument(
        "--arch",
        type=str,
        required=True,
        choices=["cnn", "trans", "mst", "ibmt_intent", "ibmt_belief", "ibmt_full"],
        help="Model architecture: cnn, trans, mst, ibmt_intent, ibmt_belief, ibmt_full",
    )
    parser.add_argument("--name", type=str, default=datetime.now().strftime("%Y%m%d_%H%M%S"), help="Run name for WandB")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    # Run trainer
    trainer = MahjongTrainer(arch=args.arch, device=args.device, run_name=args.name)
    trainer.run()

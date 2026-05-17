import argparse
import logging
from typing import Union

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.rich import tqdm

import src.config as config
from src.models import get_model, get_data_mode
from src.training.dataset import MahjongDataset
import src.utils as utils


class BeliefEvaluator:
    """
    Evaluator for Mahjong AI models (Belief prediction).
    """

    def __init__(self, runs: list[str], device: str):
        """
        Initialize evaluator components.
        """
        self.runs = runs
        self.device = utils.get_device(device)

        # Set results directory
        config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    @torch.no_grad()
    def evaluate_single(self, run_name: str) -> Union[dict, None]:
        """
        Evaluate a single model checkpoint.
        """
        # Set random seed
        utils.set_seed(config.SEED)

        # Search for the checkpoint
        model_path = None
        arch = None

        for arch_dir in config.MODEL_DIR.iterdir():
            if arch_dir.is_dir():
                potential_path = arch_dir / run_name / "best_model.pth"

                if potential_path.exists():
                    model_path = potential_path
                    break

        if model_path is None:
            logging.warning("Checkpoint '%s' not found in %s", run_name, config.MODEL_DIR)
            return None

        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        arch = checkpoint.get("arch")

        if arch is None:
            logging.error("'arch' key not found in checkpoint %s", model_path)
            return None

        logging.info("Identified architecture: %s for run: %s", arch, run_name)

        # Determine data representation
        mode = get_data_mode(arch)

        # Initialize test dataset and loader
        test_dataset = MahjongDataset(mode=mode, split="test")
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=config.NUM_PROCESSES,
            pin_memory=True,
        )

        # Build model
        model = get_model(arch).to(self.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        total_correct = 0
        total_mae = 0.0
        total_samples = 0

        pbar = tqdm(test_loader, desc=f"Evaluating Belief: {run_name}", leave=False)

        for batch in pbar:
            # Move data to target device
            states = batch["state"].to(self.device, non_blocking=True)
            opponent_shanten_numbers = batch["opponent_shanten_numbers"].to(self.device, non_blocking=True).long()

            # Forward pass
            with torch.amp.autocast("cuda"):
                output = model(states)

                if "belief" not in output:
                    continue

                logits_belief = output["belief"].view(-1, 3, config.NUM_SHANTEN_CLASSES)

            # Compute Accuracy (Top-1) and MAE
            predictions_belief = torch.argmax(logits_belief, dim=-1)

            # Count correct predictions
            total_correct += int((predictions_belief == opponent_shanten_numbers).sum().item())

            # Compute Absolute Error
            absolute_error = torch.abs(predictions_belief.float() - opponent_shanten_numbers.float())
            total_mae += float(absolute_error.sum().item())

            # Each sample has 3 opponent shanten numbers
            total_samples += opponent_shanten_numbers.numel()

        if total_samples == 0:
            logging.warning("No valid belief samples found for %s", run_name)
            return None

        return {
            "arch": arch,
            "run_name": run_name,
            "top1_acc": total_correct / total_samples,
            "mae": total_mae / total_samples,
        }

    def run(self):
        """
        Execute evaluation for all specified runs.
        """
        all_results = []

        # Evaluate each run
        for run in self.runs:
            logging.info("Starting evaluation: %s", run)
            result = self.evaluate_single(run)

            if result:
                all_results.append(result)

                logging.info("Completed evaluation: %s", run)

        if not all_results:
            logging.warning("No evaluation results were generated.")
            return

        # Create summary dataframe
        df = pd.DataFrame(all_results)

        # Merge with existing results if CSV already exists
        if config.EVAL_BELIEF_CSV.exists():
            old_df = pd.read_csv(config.EVAL_BELIEF_CSV)
            df = pd.concat([old_df, df]).drop_duplicates(subset=["arch", "run_name"], keep="last")

        # Save to CSV
        df.to_csv(config.EVAL_BELIEF_CSV, index=False)

        logging.info("Evaluation completed. Results saved to %s", config.EVAL_BELIEF_CSV)


if __name__ == "__main__":
    # Set up logging
    utils.setup_logging()

    # Parse arguments
    parser = argparse.ArgumentParser(description="Evaluate Mahjong AI models (Belief).")
    parser.add_argument(
        "--runs",
        nargs="+",
        required=True,
        help="List of run names to evaluate",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    # Run evaluation
    evaluator = BeliefEvaluator(runs=args.runs, device=args.device)
    evaluator.run()

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


class ActionEvaluator:
    """
    Evaluator for Mahjong AI models (Action prediction).
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

        total_nll = 0.0
        correct_k = {1: 0, 3: 0}
        total_samples = 0

        pbar = tqdm(test_loader, desc=f"Evaluating Action: {run_name}", leave=False)

        for batch in pbar:
            # Move data to target device
            states = batch["state"].to(self.device, non_blocking=True)
            actions = batch["action"].to(self.device, non_blocking=True)

            # Forward pass
            with torch.amp.autocast("cuda"):
                output = model(states)
                logits = output["action"]

                # Compute NLL (Negative Log Likelihood)
                nll_loss = F.cross_entropy(logits, actions, reduction="sum")

            total_nll += nll_loss.item()

            # Compute Top-K Accuracy
            _, predicted = logits.topk(3, 1, True, True)
            predicted = predicted.t()
            correct = predicted.eq(actions.view(1, -1).expand_as(predicted))

            for k in [1, 3]:
                correct_k[k] += int(correct[:k].reshape(-1).float().sum(0).item())

            total_samples += actions.size(0)

        return {
            "arch": arch,
            "run_name": run_name,
            "nll": total_nll / total_samples,
            "top1_acc": correct_k[1] / total_samples,
            "top3_acc": correct_k[3] / total_samples,
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
        if config.EVAL_ACTION_CSV.exists():
            old_df = pd.read_csv(config.EVAL_ACTION_CSV)
            df = pd.concat([old_df, df]).drop_duplicates(subset=["arch", "run_name"], keep="last")

        # Save to CSV
        df.to_csv(config.EVAL_ACTION_CSV, index=False)

        logging.info("Evaluation completed. Results saved to %s", config.EVAL_ACTION_CSV)


if __name__ == "__main__":
    # Set up logging
    utils.setup_logging()

    # Parse arguments
    parser = argparse.ArgumentParser(description="Evaluate Mahjong AI models (Action).")
    parser.add_argument(
        "--runs",
        nargs="+",
        required=True,
        help="List of run names to evaluate",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    # Run evaluation
    evaluator = ActionEvaluator(runs=args.runs, device=args.device)
    evaluator.run()

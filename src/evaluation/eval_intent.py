import argparse
import logging
from typing import Union

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.rich import tqdm

import src.config as config
from src.models import get_model, get_data_mode
from src.training.dataset import MahjongDataset
import src.utils as utils


class IntentEvaluator:
    """
    Evaluator for Mahjong AI models (Intent prediction).
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

        all_targets = []
        all_predictions = []

        pbar = tqdm(test_loader, desc=f"Evaluating Intent: {run_name}", leave=False)

        for batch in pbar:
            # Move data to target device
            states = batch["state"].to(self.device, non_blocking=True)
            yakus = batch["yaku"].to(self.device, non_blocking=True).float()
            yaku_masks = batch["yaku_mask"].to(self.device, non_blocking=True).bool()

            # Forward pass
            with torch.amp.autocast("cuda"):
                output = model(states)

                if "intent" not in output:
                    continue

                logits = output["intent"]

            # Filter samples where the player is a winner
            if yaku_masks.any():
                target_yakus = yakus[yaku_masks].cpu().numpy()
                pred_yakus = (torch.sigmoid(logits[yaku_masks]) > 0.5).float().cpu().numpy()

                all_targets.append(target_yakus)
                all_predictions.append(pred_yakus)

        if not all_targets:
            logging.warning("No valid intent samples found for %s", run_name)
            return None

        # Concatenate all results
        y_true = np.concatenate(all_targets, axis=0)
        y_pred = np.concatenate(all_predictions, axis=0)

        return {
            "arch": arch,
            "run_name": run_name,
            "exact_match_acc": accuracy_score(y_true, y_pred),
            "micro_f1": f1_score(y_true, y_pred, average="micro"),
            "macro_f1": f1_score(y_true, y_pred, average="macro"),
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
        if config.EVAL_INTENT_CSV.exists():
            old_df = pd.read_csv(config.EVAL_INTENT_CSV)
            df = pd.concat([old_df, df]).drop_duplicates(subset=["arch", "run_name"], keep="last")

        # Save to CSV
        df.to_csv(config.EVAL_INTENT_CSV, index=False)

        logging.info("Evaluation completed. Results saved to %s", config.EVAL_INTENT_CSV)


if __name__ == "__main__":
    # Set up logging
    utils.setup_logging()

    # Parse arguments
    parser = argparse.ArgumentParser(description="Evaluate Mahjong AI models (Intent).")
    parser.add_argument(
        "--runs",
        nargs="+",
        required=True,
        help="List of run names to evaluate",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    # Run evaluation
    evaluator = IntentEvaluator(runs=args.runs, device=args.device)
    evaluator.run()

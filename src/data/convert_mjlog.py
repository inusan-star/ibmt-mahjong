import logging
import multiprocessing as mp
from pathlib import Path
import subprocess

from tqdm.rich import tqdm

import src.config as config
import src.utils as utils


def _convert_single_log(mjlog_path: Path) -> None:
    """Convert a single mjlog to json."""
    json_path = config.JSON_LOGS_DIR / f"{mjlog_path.stem}.json"

    try:
        # Read mjlog with fallback encoding
        try:
            mjlog_text = mjlog_path.read_text(encoding="shift_jis")

        except UnicodeDecodeError:
            mjlog_text = mjlog_path.read_text(encoding="utf-8")

        # Run mjxc conversion command
        process = subprocess.run(
            ["mjxc", "convert", "--to-mjxproto"],
            input=mjlog_text,
            capture_output=True,
            check=True,
            text=True,
            timeout=config.SUBPROCESS_TIMEOUT,
        )

        # Save the converted json
        json_path.write_text(process.stdout.strip(), encoding="utf-8")

    except subprocess.CalledProcessError:
        logging.error("Failed to convert: %s", mjlog_path.name)

    except subprocess.TimeoutExpired:
        logging.error("Timeout converting: %s", mjlog_path.name)


def run_conversion() -> None:
    """Convert mjlogs to jsons."""
    # Ensure the output directory exists
    config.JSON_LOGS_DIR.mkdir(parents=True, exist_ok=True)

    # List all mjlog files in the input directory
    mjlog_files = sorted(list(config.MJLOG_DIR.glob("*.mjlog")))

    if not mjlog_files:
        logging.info("No mjlog files found in %s.", config.MJLOG_DIR)
        return

    logging.info("Converting %d logs...", len(mjlog_files))

    # Process each mjlog file
    with mp.Pool(processes=config.NUM_PROCESSES) as pool:
        with tqdm(total=len(mjlog_files), desc="Converting", unit="mjlog") as pbar:
            for _ in pool.imap_unordered(_convert_single_log, mjlog_files):
                pbar.update(1)

    logging.info("Successfully converted mjlogs to jsons.")


if __name__ == "__main__":
    # Set up logging
    utils.setup_logging()

    # Convert mjlogs to jsons
    run_conversion()

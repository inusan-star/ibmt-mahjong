import argparse
import json
import logging
import multiprocessing as mp
from pathlib import Path
import random
from typing import Any, Optional

import numpy as np
import mjx
from mjx import Action, Observation, State
from tqdm.rich import tqdm

import src.config as config
import src.encoders as encoders
from src.encoders.base import BaseObsEncoder
import src.utils as utils

# Global encoder
WORKER_ENCODER: Optional[BaseObsEncoder] = None


def worker_init(mode: str) -> None:
    """Init worker encoder."""
    global WORKER_ENCODER
    WORKER_ENCODER = encoders.get_encoder(mode)


def _calculate_shanten_number(hand_counts: np.ndarray, open_bits: list[int]) -> int:
    """Calculate shanten number from hand counts."""
    tile_ids = []

    for tile_type, count in enumerate(hand_counts):
        for _ in range(count):
            tile_ids.append(tile_type * 4)

    hand = mjx.Hand(json.dumps({"closedTiles": tile_ids, "opens": open_bits}))

    return hand.shanten_number()


def _get_player_open_bits(observation: Observation, player_idx: int) -> list[int]:
    """Reconstruct open bits from observation events."""
    player_open_bits = []

    for event in observation.events():
        if event.who() != player_idx:
            continue

        event_type = event.type()

        # Handle new melds
        if event_type in [
            mjx.EventType.CHI,
            mjx.EventType.PON,
            mjx.EventType.CLOSED_KAN,
            mjx.EventType.OPEN_KAN,
        ]:
            player_open_bits.append(event.open().bit)

        # Handle added kan
        elif event_type == mjx.EventType.ADDED_KAN:
            added_tile_type = event.open().last_tile().type()

            for i, open_bit in enumerate(player_open_bits):
                potential_pon = mjx.Open(open_bit)

                if (
                    potential_pon.event_type() == mjx.EventType.PON
                    and potential_pon.stolen_tile().type() == added_tile_type
                ):
                    player_open_bits[i] = event.open().bit
                    break

    return player_open_bits


def _get_post_action_hand(observation: Observation, action: Action) -> np.ndarray:
    """Calculate hand counts after the action."""
    hand_counts = np.zeros(34, dtype=np.int8)

    for tile_type, count in enumerate(observation.curr_hand().closed_tile_types()):
        hand_counts[tile_type] = count

    action_idx = action.to_idx()

    # Discard
    if action_idx <= 36:
        tile_type = action_idx if action_idx < 34 else {34: 4, 35: 13, 36: 22}[action_idx]
        hand_counts[tile_type] -= 1

    # Tsumogiri
    elif action_idx <= 73:
        tmp_action_idx = action_idx - 37
        tile_type = tmp_action_idx if tmp_action_idx < 34 else {34: 4, 35: 13, 36: 22}[tmp_action_idx]
        hand_counts[tile_type] -= 1

    # Chi
    elif action_idx <= 103:
        if action_idx <= 80:
            start_type = action_idx - 74

        elif action_idx <= 87:
            start_type = (action_idx - 81) + 9

        elif action_idx <= 94:
            start_type = (action_idx - 88) + 18

        elif action_idx <= 97:
            start_type = (action_idx - 95) + 2

        elif action_idx <= 100:
            start_type = (action_idx - 98) + 11

        else:
            start_type = (action_idx - 101) + 20

        called_tile_type = next(
            event.tile().type()
            for event in reversed(observation.events())
            if event.type() in (mjx.EventType.DISCARD, mjx.EventType.TSUMOGIRI)
        )

        for tile_type in range(start_type, start_type + 3):
            if tile_type != called_tile_type:
                hand_counts[tile_type] -= 1

    # Pon
    elif action_idx <= 140:
        called_tile_type = next(
            event.tile().type()
            for event in reversed(observation.events())
            if event.type() in (mjx.EventType.DISCARD, mjx.EventType.TSUMOGIRI)
        )
        hand_counts[called_tile_type] -= 2

    # Kan
    elif action_idx <= 174:
        hand_counts[action_idx - 141] = 0

    return hand_counts


def _process_round(round_line: str) -> list[dict[str, Any]]:
    """Process round features."""
    if WORKER_ENCODER is None:
        return []

    try:
        # Init state
        state_json = round_line.strip()
        state = State(state_json)
        raw_data = json.loads(state_json)
        results = []

        # Extract yaku labels per winner
        terminal = state.to_proto().round_terminal
        player_yaku = {}

        if terminal and terminal.wins:
            for win in terminal.wins:
                yaku_labels = np.zeros(config.NUM_YAKU, dtype=np.bool_)

                for yaku_index in list(win.yakus) + list(win.yakumans):
                    if yaku_index < config.NUM_YAKU:
                        yaku_labels[yaku_index] = True

                player_yaku[win.who] = yaku_labels

        # Track latest hand counts for each player
        latest_hands = [np.zeros(34, dtype=np.int8) for _ in range(4)]

        for private_obs in raw_data["privateObservations"]:
            player_idx = private_obs.get("who", 0)

            for tile_id in private_obs["initHand"]["closedTiles"]:
                latest_hands[player_idx][tile_id // 4] += 1

        # Process decisions
        for cpp_observation, cpp_action in state._cpp_obj.past_decisions():
            observation = Observation._from_cpp_obj(cpp_observation)
            action = Action._from_cpp_obj(cpp_action)

            # Filter discards
            action_id = action.to_idx()
            action_index = BaseObsEncoder.normalize_action_id(action_id)

            if BaseObsEncoder.is_discard_action(action_index):
                # Encode feature
                feature = WORKER_ENCODER.encode(observation)

                # Check if this player is a winner
                current_player = observation.who()
                is_winner = current_player in player_yaku

                # Encode opponent shanten numbers labels
                opponent_shanten_numbers_labels = np.zeros(3, dtype=np.int8)

                for i in range(1, 4):
                    opponent_abs_idx = (current_player + i) % 4
                    opponent_shanten_numbers_labels[i - 1] = _calculate_shanten_number(
                        latest_hands[opponent_abs_idx], _get_player_open_bits(observation, opponent_abs_idx)
                    )

                results.append(
                    {
                        "state": feature,
                        "action": action_index,
                        "yaku": player_yaku.get(current_player, np.zeros(config.NUM_YAKU, dtype=np.bool_)),
                        "yaku_mask": is_winner,
                        "opponent_shanten_numbers": opponent_shanten_numbers_labels,
                    }
                )

            # Update acting player's hand counts
            latest_hands[observation.who()] = _get_post_action_hand(observation, action)

        return results

    except Exception as e:
        logging.warning("Failed to process round: %s", e)
        return []


def _save_chunk(data_list: list[dict], split: str, chunk_index: int, mode: str) -> None:
    """Save data chunk."""
    # Set save directory
    save_dir = config.PROCESSED_DIR / mode / split
    save_dir.mkdir(parents=True, exist_ok=True)

    # Convert to arrays
    states = np.array([item["state"] for item in data_list], dtype=np.bool_)
    actions = np.array([item["action"] for item in data_list], dtype=np.int64)
    yakus = np.array([item["yaku"] for item in data_list], dtype=np.bool_)
    yaku_masks = np.array([item["yaku_mask"] for item in data_list], dtype=np.bool_)
    opponent_shanten_numbers = np.array([item["opponent_shanten_numbers"] for item in data_list], dtype=np.int8)

    # Save files
    np.save(save_dir / f"state_{chunk_index:03d}.npy", states)
    np.save(save_dir / f"action_{chunk_index:03d}.npy", actions)
    np.save(save_dir / f"yaku_{chunk_index:03d}.npy", yakus)
    np.save(save_dir / f"yaku_mask_{chunk_index:03d}.npy", yaku_masks)
    np.save(save_dir / f"opponent_shanten_numbers_{chunk_index:03d}.npy", opponent_shanten_numbers)


def _load_rounds_from_files(file_list: list[Path]) -> list[str]:
    """Load round lines from files."""
    all_rounds = []

    for file_path in file_list:
        with open(file_path, "r", encoding="utf-8") as file_handle:
            all_rounds.extend(file_handle.readlines())

    return all_rounds


def _pick_one_round_per_file(file_list: list[Path]) -> list[str]:
    """Pick one random round from each file."""
    picked_rounds = []

    for file_path in file_list:
        with open(file_path, "r", encoding="utf-8") as file_handle:
            content = file_handle.readlines()

            if content:
                picked_rounds.append(random.choice(content))

    return picked_rounds


def _get_split_configurations() -> dict[str, tuple[list[str], int, bool]]:
    """Split data by game-unit."""
    # List game files
    game_files = sorted(
        [file_path for file_path in config.JSON_LOGS_DIR.rglob("*.json") if file_path.stat().st_size > 0]
    )
    logging.info("Found %d game files.", len(game_files))

    # Shuffle games
    random.shuffle(game_files)

    # Split games (Pick 1.1x rounds for single sampling to ensure enough data)
    valid_games = game_files[: int(config.TARGET_VALID_DATA * 1.1)]
    test_games = game_files[len(valid_games) : len(valid_games) + int(config.TARGET_TEST_DATA * 1.1)]
    train_games = game_files[len(valid_games) + len(test_games) :]

    return {
        "valid": (_pick_one_round_per_file(valid_games), config.TARGET_VALID_DATA, True),
        "test": (_pick_one_round_per_file(test_games), config.TARGET_TEST_DATA, True),
        "train": (_load_rounds_from_files(train_games), config.TARGET_TRAIN_DATA, False),
    }


def build_dataset(mode: str) -> None:
    """Run dataset building."""
    # Set random seed
    utils.set_seed(config.SEED)

    # Get split configurations
    split_configurations = _get_split_configurations()

    # Process pool
    with mp.Pool(
        processes=config.NUM_PROCESSES,
        initializer=worker_init,
        initargs=(mode,),
    ) as pool:
        for split_name, (rounds, target_count, is_single_sampling) in split_configurations.items():
            logging.info("Processing %s split (Mode: %s)", split_name, mode)

            buffer = []
            total_extracted = 0
            chunk_index = 0

            # Progress bar
            with tqdm(total=target_count, desc=split_name, unit="sample") as pbar:
                batch_size = config.PARALLEL_BATCH_SIZE

                for i in range(0, len(rounds), batch_size):
                    rounds_batch = rounds[i : i + batch_size]

                    for round_results in pool.imap_unordered(_process_round, rounds_batch):
                        # Pick only one situation if it's valid/test
                        if is_single_sampling:
                            if not round_results:
                                continue

                            data_to_add = [random.choice(round_results)]

                        else:
                            data_to_add = round_results

                        # Add data to buffer
                        for data in data_to_add:
                            if total_extracted >= target_count:
                                break

                            buffer.append(data)
                            total_extracted += 1
                            pbar.update(1)

                            # Save chunk
                            if len(buffer) >= config.TRAIN_CHUNK_SIZE:
                                _save_chunk(buffer, split_name, chunk_index, mode)
                                buffer = []
                                chunk_index += 1

                        if total_extracted >= target_count:
                            break

                    if total_extracted >= target_count:
                        break

                # Save remaining
                if buffer:
                    _save_chunk(buffer, split_name, chunk_index, mode)

    logging.info("Successfully created dataset for %s.", mode)


if __name__ == "__main__":
    # Set up logging
    utils.setup_logging()

    # Parse argsuments
    parser = argparse.ArgumentParser(description="Build dataset.")
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["cnn", "symbolic"],
        help="Target representation: 'cnn' or 'symbolic'",
    )
    args = parser.parse_args()

    # Run dataset building
    build_dataset(args.mode)

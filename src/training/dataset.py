import numpy as np
import torch
from torch.utils.data import Dataset

import src.config as config


class MahjongDataset(Dataset):
    """
    Dataset for Mahjong training.
    """

    def __init__(self, mode: str, split: str):
        self.data_dir = config.PROCESSED_DIR / mode / split

        # Collect chunk IDs
        chunk_ids = sorted([int(path.stem.split("_")[1]) for path in self.data_dir.glob("state_*.npy")])

        if not chunk_ids:
            raise FileNotFoundError(f"No data in {self.data_dir}")

        # Memory-mapped arrays
        self.state_list: list[np.ndarray] = []
        self.action_list: list[np.ndarray] = []
        self.yaku_list: list[np.ndarray] = []
        self.yaku_mask_list: list[np.ndarray] = []
        self.opponent_shanten_numbers_list: list[np.ndarray] = []

        chunk_sizes = []
        current_total = 0

        for chunk_id in chunk_ids:
            # Map files to handles
            state_path = self.data_dir / f"state_{chunk_id:03d}.npy"
            action_path = self.data_dir / f"action_{chunk_id:03d}.npy"
            yaku_path = self.data_dir / f"yaku_{chunk_id:03d}.npy"
            yaku_mask_path = self.data_dir / f"yaku_mask_{chunk_id:03d}.npy"
            opponent_shanten_numbers_path = self.data_dir / f"opponent_shanten_numbers_{chunk_id:03d}.npy"

            states = np.load(state_path, mmap_mode="r")
            actions = np.load(action_path, mmap_mode="r")
            yakus = np.load(yaku_path, mmap_mode="r")
            yaku_masks = np.load(yaku_mask_path, mmap_mode="r")
            opponent_shanten_numbers = np.load(opponent_shanten_numbers_path, mmap_mode="r")

            self.state_list.append(states)
            self.action_list.append(actions)
            self.yaku_list.append(yakus)
            self.yaku_mask_list.append(yaku_masks)
            self.opponent_shanten_numbers_list.append(opponent_shanten_numbers)

            chunk_size = states.shape[0]
            chunk_sizes.append(chunk_size)
            current_total += chunk_size

        self.total_samples = current_total

        self.idx_to_chunk = np.empty(self.total_samples, dtype=np.int16)
        self.idx_to_local = np.empty(self.total_samples, dtype=np.int32)

        start = 0
        
        for i, chunk_size in enumerate(chunk_sizes):
            end = start + chunk_size
            self.idx_to_chunk[start:end] = i
            self.idx_to_local[start:end] = np.arange(chunk_size)
            start = end

    def __len__(self) -> int:
        return self.total_samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        chunk_idx = self.idx_to_chunk[idx]
        local_idx = self.idx_to_local[idx]

        # Fetch data
        state = self.state_list[chunk_idx][local_idx]
        action = int(self.action_list[chunk_idx][local_idx])
        yaku = self.yaku_list[chunk_idx][local_idx]
        yaku_mask = bool(self.yaku_mask_list[chunk_idx][local_idx])
        opponent_shanten_numbers = self.opponent_shanten_numbers_list[chunk_idx][local_idx]

        return {
            "state": torch.from_numpy(state.copy()).float(),
            "action": torch.tensor(action, dtype=torch.long),
            "yaku": torch.from_numpy(yaku.copy()).float(),
            "yaku_mask": torch.tensor(yaku_mask, dtype=torch.float32),
            "opponent_shanten_numbers": torch.from_numpy(opponent_shanten_numbers.copy()).long(),
        }
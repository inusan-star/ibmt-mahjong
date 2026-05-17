from abc import ABC, abstractmethod

import mjx
import numpy as np


class BaseObsEncoder(ABC):
    """Base class for observation encoders."""

    @abstractmethod
    def encode(self, obs: mjx.Observation) -> np.ndarray:
        """Encode observation into a feature array."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for the encoder."""

    @property
    @abstractmethod
    def shape(self) -> tuple:
        """Shape of the encoded feature."""

    @staticmethod
    def normalize_action_id(action_id: int) -> int:
        """Normalize action IDs."""
        # Map tsumogiri (37-73) -> discard (0-33)
        if 37 <= action_id <= 73:
            action_id -= 37

        # Map red tiles (34-36, 95-103, 138-140) to standard tiles
        if 34 <= action_id <= 36:
            action_id = 4 + (action_id - 34) * 9

        elif 95 <= action_id <= 97:
            action_id -= 19

        elif 98 <= action_id <= 100:
            action_id -= 15

        elif 101 <= action_id <= 103:
            action_id -= 11

        elif 138 <= action_id <= 140:
            action_id = 108 + (action_id - 138) * 9

        return action_id

    @staticmethod
    def is_discard_action(action_id: int) -> bool:
        """Check if the action is a discard."""
        return 0 <= action_id <= 33

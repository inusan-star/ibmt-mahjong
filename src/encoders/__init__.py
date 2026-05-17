from src.encoders.base import BaseObsEncoder
from src.encoders.cnn_encoder import CNNObsEncoder
from src.encoders.symbolic_encoder import SymbolicObsEncoder


def get_encoder(mode: str, **kwargs) -> BaseObsEncoder:
    """Factory to get an encoder instance."""
    if mode == "cnn":
        return CNNObsEncoder(**kwargs)

    if mode == "symbolic":
        return SymbolicObsEncoder(**kwargs)

    raise ValueError(f"Unknown mode: {mode}.")

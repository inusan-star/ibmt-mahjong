import torch.nn as nn

import src.config as config
from src.models.cnn import CNN
from src.models.trans import Transformer
from src.models.mst import MST
from src.models.ibmt_intent import IBMTIntent
from src.models.ibmt_belief import IBMTBelief
from src.models.ibmt_full import IBMTFull


def get_model(arch: str) -> nn.Module:
    """
    Get model architecture.
    """
    if arch == "cnn":
        return CNN(
            image_size=config.CNN_IMAGE_SIZE,
            channels=config.CNN_CHANNELS,
            num_classes=config.CNN_NUM_CLASSES,
            num_block=config.CNN_NUM_BLOCK,
            hidden_dim=config.CNN_HIDDEN_DIM,
            kernel_size=config.CNN_KERNEL_SIZE,
            stride=config.CNN_STRIDE,
            padding=config.CNN_PADDING,
        )

    elif arch == "trans":
        return Transformer(
            seq_len=config.TRANS_SEQ_LEN,
            input_dim=config.TRANS_INPUT_DIM,
            num_classes=config.TRANS_NUM_CLASSES,
            emb_dim=config.TRANS_EMB_DIM,
            num_heads=config.TRANS_NUM_HEADS,
            num_layers=config.TRANS_NUM_LAYERS,
            dim_feedforward=config.TRANS_DIM_FEEDFORWARD,
            dropout=config.TRANS_DROPOUT,
        )

    elif arch == "mst":
        return MST(
            seq_len=config.MST_SEQ_LEN,
            input_dim=config.MST_INPUT_DIM,
            num_classes=config.MST_NUM_CLASSES,
            emb_dim=config.MST_EMB_DIM,
            num_heads=config.MST_NUM_HEADS,
            num_layers=config.MST_NUM_LAYERS,
            dim_feedforward=config.MST_DIM_FEEDFORWARD,
            dropout=config.MST_DROPOUT,
        )

    elif arch == "ibmt_intent":
        return IBMTIntent(
            seq_len=config.IBMT_INTENT_SEQ_LEN,
            input_dim=config.IBMT_INTENT_INPUT_DIM,
            num_classes=config.IBMT_INTENT_NUM_CLASSES,
            emb_dim=config.IBMT_INTENT_EMB_DIM,
            num_heads=config.IBMT_INTENT_NUM_HEADS,
            num_layers=config.IBMT_INTENT_NUM_LAYERS,
            dim_feedforward=config.IBMT_INTENT_DIM_FEEDFORWARD,
            dropout=config.IBMT_INTENT_DROPOUT,
        )

    elif arch == "ibmt_belief":
        return IBMTBelief(
            seq_len=config.IBMT_BELIEF_SEQ_LEN,
            input_dim=config.IBMT_BELIEF_INPUT_DIM,
            num_classes=config.IBMT_BELIEF_NUM_CLASSES,
            emb_dim=config.IBMT_BELIEF_EMB_DIM,
            num_heads=config.IBMT_BELIEF_NUM_HEADS,
            num_layers=config.IBMT_BELIEF_NUM_LAYERS,
            dim_feedforward=config.IBMT_BELIEF_DIM_FEEDFORWARD,
            dropout=config.IBMT_BELIEF_DROPOUT,
        )

    elif arch == "ibmt_full":
        return IBMTFull(
            seq_len=config.IBMT_FULL_SEQ_LEN,
            input_dim=config.IBMT_FULL_INPUT_DIM,
            num_classes=config.IBMT_FULL_NUM_CLASSES,
            emb_dim=config.IBMT_FULL_EMB_DIM,
            num_heads=config.IBMT_FULL_NUM_HEADS,
            num_layers=config.IBMT_FULL_NUM_LAYERS,
            dim_feedforward=config.IBMT_FULL_DIM_FEEDFORWARD,
            dropout=config.IBMT_FULL_DROPOUT,
        )

    raise NotImplementedError(f"Architecture '{arch}' not implemented.")


def get_data_mode(arch: str) -> str:
    """
    Get data representation mode.
    """
    modes = {
        "cnn": "cnn",
        "trans": "symbolic",
        "mst": "symbolic",
        "ibmt_intent": "symbolic",
        "ibmt_belief": "symbolic",
        "ibmt_full": "symbolic",
    }

    if arch not in modes:
        raise NotImplementedError(f"Architecture '{arch}' not implemented.")

    return modes[arch]

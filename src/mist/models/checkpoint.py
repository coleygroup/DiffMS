"""Load coleygroup/MIST EMA checkpoints for spectrum-to-fingerprint inference."""

from pathlib import Path

import torch

from .scaled import ScaledMIST


def load_mist_checkpoint(path, device="cpu"):
    """Construct the model from its saved configuration and load all EMA weights."""
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or not {"ema_model", "config"} <= checkpoint.keys():
        raise ValueError("Expected a coleygroup/MIST checkpoint with config and ema_model")
    config = checkpoint["config"].get("model")
    if not isinstance(config, dict):
        raise ValueError("MIST checkpoint is missing config.model")
    model = ScaledMIST(**config)
    model.load_state_dict(checkpoint["ema_model"], strict=True)
    model.model_config = dict(config)
    model.dataset_name = checkpoint["config"].get("data", {}).get("dataset_name")
    return model.to(device).eval()


def configure_encoder_finetuning(encoder, strategy):
    """Select MIST transformer or fingerprint-head parameters for downstream tuning."""
    if strategy is None:
        return
    if strategy not in {"freeze", "ft-unfold", "freeze-unfold", "ft-transformer", "freeze-transformer"}:
        raise ValueError(f"Unknown encoder strategy: {strategy}")
    scaled = isinstance(encoder, ScaledMIST)
    for name, parameter in encoder.named_parameters():
        head = name.startswith("decoder." if scaled else "spectra_encoder.2.")
        transformer = name.startswith("encoder." if scaled else "spectra_encoder.0.")
        freeze = (
            strategy == "freeze"
            or (strategy == "ft-unfold" and not head)
            or (strategy == "freeze-unfold" and head)
            or (strategy == "ft-transformer" and not transformer)
            or (strategy == "freeze-transformer" and transformer)
        )
        if freeze:
            parameter.requires_grad = False

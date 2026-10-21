"""A lean, explicitly constructed MIST encoder for native PyTorch training."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .modules import FormulaTransformer, FPGrowingModule


class ScaledMIST(nn.Module):
    """Formula Transformer plus iterative fingerprint unfolding.

    This preserves the two strongest MIST inductive biases—formula attention
    and coarse-to-fine fingerprint prediction—while omitting the unused MAGMa
    fragment head from the historical training graph.
    """

    def __init__(
        self,
        *,
        hidden_size: int = 768,
        peak_attn_layers: int = 3,
        num_heads: int = 12,
        refine_layers: int = 4,
        spectra_dropout: float = 0.1,
        pairwise_featurization: bool = True,
        no_diffs: bool = True,
        form_embedder: str = "pos-cos",
        output_size: int = 4096,
        set_pooling: str = "cls",
        embed_instrument: bool = False,
        **unused: Any,
    ):
        super().__init__()
        if hidden_size % num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        if output_size != 4096:
            raise ValueError("MIST predicts a 4096-bit Morgan fingerprint")
        if refine_layers < 0:
            raise ValueError("refine_layers must be nonnegative")
        self.hidden_size = int(hidden_size)
        self.output_size = int(output_size)
        self.refine_layers = int(refine_layers)
        self.encoder = FormulaTransformer(
            hidden_size=hidden_size,
            peak_attn_layers=peak_attn_layers,
            num_heads=num_heads,
            spectra_dropout=spectra_dropout,
            pairwise_featurization=pairwise_featurization,
            no_diffs=no_diffs,
            form_embedder=form_embedder,
            set_pooling=set_pooling,
            embed_instrument=embed_instrument,
            inten_transform="float",
            output_size=output_size,
        )
        self.decoder = FPGrowingModule(
            hidden_input_dim=hidden_size,
            final_target_dim=output_size,
            num_splits=refine_layers,
            reduce_factor=2,
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict]:
        encoded = self.encoder(batch)
        predictions = self.decoder(encoded)
        return predictions[-1], {
            "intermediate_predictions": predictions[:-1],
            "int_preds": predictions[:-1],
            "h0": encoded,
            "prediction_dims": self.decoder.output_dims,
        }


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())

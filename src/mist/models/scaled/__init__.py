"""Inference model from coleygroup/MIST (8c03c5d), distributed under MIT.

The auxiliary h0/int_preds aliases expose the downstream encoder interface.
"""

from .scaled_mist import ScaledMIST

__all__ = ["ScaledMIST"]

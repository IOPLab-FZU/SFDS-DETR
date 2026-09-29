"""Backward-compatible import for checkpoints created before paper alignment."""

from .ospfe import OSRF

RepC2f_E = OSRF

__all__ = ("RepC2f_E",)

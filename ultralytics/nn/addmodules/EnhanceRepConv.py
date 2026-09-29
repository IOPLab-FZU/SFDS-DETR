"""Backward-compatible import for checkpoints created before paper alignment."""

from .ospfe import OSRConv

EnhancedRepConv = OSRConv

__all__ = ("EnhancedRepConv",)

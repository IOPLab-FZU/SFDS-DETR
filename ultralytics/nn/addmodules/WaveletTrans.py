"""Backward-compatible imports for checkpoints created before paper alignment."""

from .fafeb import CascadedWaveletTransform, WaveletTransform

HaarWaveletTrans = WaveletTransform
HaarWaveletTrans_fromyL = CascadedWaveletTransform

__all__ = ("HaarWaveletTrans", "HaarWaveletTrans_fromyL")

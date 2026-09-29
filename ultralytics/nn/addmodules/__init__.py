"""Paper-aligned modules required by SFDS-DETR."""

from .fafeb import CAAE, CEM, SE, CascadedWaveletTransform, ConvBlock, WaveletTransform
from .ospfe import OSRF, OSRConv

# Legacy aliases keep older YAML files and pickled model classes importable.
EnhancedRepConv = OSRConv
HaarWaveletTrans = WaveletTransform
HaarWaveletTrans_fromyL = CascadedWaveletTransform
MFM = CEM
RepC2f_E = OSRF
SELayer = SE

__all__ = (
    "CAAE",
    "CEM",
    "CascadedWaveletTransform",
    "ConvBlock",
    "OSRF",
    "OSRConv",
    "SE",
    "WaveletTransform",
    "EnhancedRepConv",
    "HaarWaveletTrans",
    "HaarWaveletTrans_fromyL",
    "MFM",
    "RepC2f_E",
    "SELayer",
)

'''
Models for encoding and decoding neural data, such as
Temporal Receptive Fields (TRFs).
'''

from .trf import TRF
from .banded_trf import BandedTRF

__all__ = ['TRF', 'BandedTRF']
# Copyright (c) 2026 Adaptive Cochlear Processor Team
# Licensed under the MIT License. See LICENSE file for details.

"""
ACP Backend Module
Pluggable decoder backends for acoustic transcription.
"""

from .interface import DecoderBackend, WhisperDecoder, MinimalDecoder

__all__ = ["DecoderBackend", "WhisperDecoder", "MinimalDecoder"]

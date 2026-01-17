# Copyright (c) 2026 Adaptive Cochlear Processor Team
# Licensed under the MIT License. See LICENSE file for details.

"""
Router Configuration - Explicit Fallback Triggers
"""
import os
import platform  # ← NEW: cross-platform safety

class RouterConfig:
    """
    All thresholds are deterministic. No magic.
    """
    
    # === STT ROUTER RULES ===
    STT_CONFIDENCE_THRESHOLD = 0.45  # Below this: fallback
    STT_LATENCY_THRESHOLD_MS = 5000  # Slower than this: fallback
    STT_MAX_RETRIES = 1              # ACP gets one chance
    
    # === TTS ROUTER RULES ===
    TTS_LATENCY_THRESHOLD_MS = 8000
    TTS_QUALITY_THRESHOLD = 0.6      # Subjective quality floor
    
    # === GLOBAL SWITCHES ===
    # Set via environment: ACP_ENABLED=0 bypasses ACP entirely
    ACP_ENABLED = os.getenv("ACP_ENABLED", "1") == "1"
    
    # WSL detection that works everywhere
    IS_WSL = "microsoft" in platform.release().lower()  # ← NEW: no os.uname()
    
    # Mode flags
    TRAINING_MODE = os.getenv("ACP_TRAINING", "0") == "1"  # If training, always use fallbacks for truth
    
    @classmethod
    def get_status(cls):
        """Current router state for monitoring"""
        return {
            "acp_enabled": cls.ACP_ENABLED,
            "is_wsl": cls.IS_WSL,
            "training_mode": cls.TRAINING_MODE,
            "stt_thresholds": {
                "confidence": cls.STT_CONFIDENCE_THRESHOLD,
                "latency_ms": cls.STT_LATENCY_THRESHOLD_MS
            }
        }

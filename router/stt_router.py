# Copyright (c) 2026 Adaptive Cochlear Processor Team
# Licensed under the MIT License. See LICENSE file for details.

"""
STT Router: Routes audio → ACP or Whisper based on deterministic rules.
ACP never knows it was bypassed. Whisper never knows ACP exists.
"""
import time
import logging
from collections import Counter  # ← NEW
from typing import Dict, Any
from pathlib import Path

# Import ACP (will fail gracefully on Windows)
try:
    from acp_1_0.acp.core import AdaptiveCochlearProcessor
    ACP_AVAILABLE = True
except ImportError:
    ACP_AVAILABLE = False
    AdaptiveCochlearProcessor = None

# Import Whisper fallback (always available)
try:
    from whisper_baseline.transcribe import WhisperBaseline
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    WhisperBaseline = None

from .config import RouterConfig

logger = logging.getLogger(__name__)

class STTRouter:
    """
    Single method: transcribe(audio_path) → result dict
    Guarantees: always returns a valid transcript, never raises
    """
    
    def __init__(self):
        self.config = RouterConfig()
        self.acp = None
        self.whisper = None
        self.fallback_was_used = False
        self.metrics = Counter()
        
        # Initialize ACP Core only if enabled and in WSL2
        if self.config.ACP_ENABLED and self.config.IS_WSL and ACP_AVAILABLE:
            try:
                # Use real ACP core with persistent SKG
                self.acp = AdaptiveCochlearProcessor(
                    skg_path="skg/hearing.json",
                    decoder=None  # Uses default WhisperDecoder for now
                )
                logger.info("✅ ACP 1.0 Core initialized as primary STT")
            except Exception as e:
                logger.warning(f"⚠️ ACP Core failed to initialize: {e}. Fallback only.")
                self.acp = None
        else:
            if not self.config.IS_WSL:
                logger.info("🏁 Running on Windows: ACP disabled, Whisper only")
            elif not self.config.ACP_ENABLED:
                logger.info("🚫 ACP disabled via config: Whisper only")
            elif not ACP_AVAILABLE:
                logger.warning("⚠️ ACP import failed. Check installation.")
        
        # Initialize Whisper fallback
        if WHISPER_AVAILABLE:
            try:
                self.whisper = WhisperBaseline()
                logger.info("✅ Whisper baseline initialized")
            except Exception as e:
                logger.warning(f"⚠️ Whisper baseline failed to initialize: {e}")
                self.whisper = None
        else:
            logger.warning("⚠️ Whisper baseline import failed")
    
    def transcribe(self, audio_path: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Main entry point. Returns transcript + metadata.
        ACP errors are caught and logged but never propagated to caller.
        """
        start_time = time.time()
        
        # === DECISION LOGIC ===
        use_acp = (
            self.acp is not None 
            and not self.config.TRAINING_MODE  # Training uses ground truth
        )
        
        if use_acp:
            try:
                # Attempt ACP with timeout
                result = self._transcribe_with_acp(audio_path, context)
                
                # Validate result
                if self._is_valid_result(result):
                    # Check confidence
                    if result["confidence"] >= self.config.STT_CONFIDENCE_THRESHOLD:
                        # Check latency
                        latency = (time.time() - start_time) * 1000
                        if latency <= self.config.STT_LATENCY_THRESHOLD_MS:
                            self.metrics["acp_calls"] += 1  # ← NEW
                            self.metrics["total_latency"] += latency
                            logger.info(
                                f"✅ ACP success: {len(result['transcript'])} chars, "
                                f"conf={result['confidence']:.2f}, "
                                f"corrections={len(result.get('corrections', []))}"
                            )
                            return result
                        else:
                            logger.warning(f"⏱️ ACP too slow ({latency:.0f}ms), falling back")
                    else:
                        logger.warning(f"📉 ACP confidence too low ({result['confidence']:.2f}), falling back")
                else:
                    logger.warning("⚠️ ACP returned malformed result, falling back")
                    
            except Exception as e:
                logger.error(f"🔥 ACP Core crashed: {e}. Activating Whisper fallback.", exc_info=True)
        
        # === FALLBACK PATH ===
        # If we get here, either ACP is disabled or failed
        self.fallback_was_used = True
        self.metrics["fallback_calls"] += 1  # ← NEW
        try:
            if self.whisper is None:
                raise RuntimeError("Whisper baseline not available")
            fallback_result = self.whisper.transcribe(audio_path)
            logger.info(f"🏁 Whisper fallback: {len(fallback_result['transcript'])} chars")
            
            # Decorate with metadata to show it was fallback
            return {
                **fallback_result,
                "_source": "whisper_baseline",
                "_acp_failed": True,
                "_timestamp": time.time()
            }
            
        except Exception as e:
            # This should never happen - Whisper is the safety net
            logger.critical(f"🚨 Whisper fallback also failed: {e}", exc_info=True)
            return {
                "transcript": "",
                "confidence": 0.0,
                "error": "both_acp_and_whisper_failed",
                "_timestamp": time.time()
            }
            return {
                "transcript": "",
                "confidence": 0.0,
                "error": "both_acp_and_whisper_failed",
                "_timestamp": time.time()
            }
    
    def _transcribe_with_acp(self, audio_path: str, context: dict) -> Dict[str, Any]:
        """Call real ACP Core (not placeholder adapter)"""
        # ACP.hear() returns format that matches router contract directly
        return self.acp.hear(audio_path, context)
    
    def _is_valid_result(self, result: Dict) -> bool:
        """Hard validation - no guesswork"""
        if not isinstance(result, dict):
            return False
        
        required_keys = {"transcript", "confidence"}
        if not required_keys.issubset(result.keys()):
            return False
        
        if not isinstance(result["transcript"], str):
            return False
        
        if not (0.0 <= result["confidence"] <= 1.0):
            return False
        
        return True
    
    def get_health_status(self) -> Dict[str, Any]:
        """Monitoring endpoint"""
        return {
            "acp_available": self.acp is not None,
            "acp_enabled": self.config.ACP_ENABLED,
            "is_wsl": self.config.IS_WSL,
            "whisper_available": self.whisper is not None,
            "training_mode": self.config.TRAINING_MODE,
            "fallback_was_used": self.fallback_was_used,
            "metrics": dict(self.metrics)
        }
    
    def get_metrics(self) -> Dict[str, float]:  # ← NEW METHOD
        """Return performance stats"""
        total = self.metrics["acp_calls"] + self.metrics["fallback_calls"]
        if total == 0:
            return {"acp_success_rate": 0.0, "mean_latency_ms": 0.0}
        
        return {
            "acp_success_rate": self.metrics["acp_calls"] / total,
            "fallback_rate": self.metrics["fallback_calls"] / total,
            "mean_latency_ms": self.metrics["total_latency"] / self.metrics["acp_calls"]
        }

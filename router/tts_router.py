# Copyright (c) 2026 Adaptive Cochlear Processor Team
# Licensed under the MIT License. See LICENSE file for details.

"""
TTS Router: Routes text → POM (via ACP) or Kokoro baseline fallback.
Same pattern as STT router: ACP never knows it was bypassed.
"""
import time
import logging
from pathlib import Path
from typing import Dict, Any

# Import ACP's voice module (WSL2 only)
try:
    from acp_1_0.acp.voice import ACPSynthesisEngine
    ACP_VOICE_AVAILABLE = True
except ImportError:
    ACP_VOICE_AVAILABLE = False
    ACPSynthesisEngine = None

# Import Kokoro baseline fallback (Windows-friendly)
from kokoro_baseline.synthesize import KokoroBaseline

from .config import RouterConfig

logger = logging.getLogger(__name__)

class TTSRouter:
    """
    Single method: synthesize(text, speaker_id) → audio_path
    Guarantees: always returns valid audio, never raises
    """
    
    def __init__(self):
        self.config = RouterConfig()
        self.acp_voice = None
        self.kokoro = KokoroBaseline()
        self.fallback_was_used = False
        
        # Initialize ACP voice only in WSL2
        if self.config.ACP_ENABLED and self.config.IS_WSL and ACP_VOICE_AVAILABLE:
            try:
                # ACP voice gets learning from SKG (hearing errors adjust speech)
                self.acp_voice = ACPSynthesisEngine(skg_path="skg/hearing.json")
                logger.info("✅ ACP Voice initialized as primary TTS")
            except Exception as e:
                logger.warning(f"⚠️ ACP Voice failed: {e}. Kokoro baseline only.")
                self.acp_voice = None
        else:
            logger.info("🏁 Kokoro baseline as primary fallback TTS (ACP voice disabled)")
    
    def synthesize(self, text: str, speaker_id: str = None, **kwargs) -> Dict[str, Any]:
        """
        Main entry point. Returns audio path + metadata.
        ACP voice errors never propagate to caller.
        """
        start_time = time.time()
        
        # === DECISION LOGIC ===
        use_acp_voice = (
            self.acp_voice is not None
            and not self.config.TRAINING_MODE
        )
        
        if use_acp_voice:
            try:
                result = self._synthesize_with_acp(text, speaker_id, **kwargs)
                
                if self._is_valid_audio(result):
                    latency = (time.time() - start_time) * 1000
                    if latency <= self.config.TTS_LATENCY_THRESHOLD_MS:
                        logger.info(f"✅ ACP Voice: {len(text)} chars, latency={latency:.0f}ms")
                        return result
                    else:
                        logger.warning(f"⏱️ ACP Voice too slow ({latency:.0f}ms), falling back")
                else:
                    logger.warning("⚠️ ACP Voice produced invalid audio, falling back")
                    
            except Exception as e:
                logger.error(f"🔥 ACP Voice crashed: {e}. Activating Kokoro baseline.")

        # === FALLBACK PATH ===
        self.fallback_was_used = True
        try:
            fallback_result = self.kokoro.synthesize(text, speaker_id)
            logger.info(f"🏁 Kokoro baseline fallback: {len(text)} chars")
            
            return {
                **fallback_result,
                "_source": "kokoro_baseline",
                "_acp_voice_failed": True,
                "_timestamp": time.time()
            }
            
        except Exception as e:
            logger.critical(f"🚨 Kokoro baseline fallback also failed: {e}")
            # Return silent audio as last resort
            return {
                "audio_path": "fallback_silence.wav",
                "error": "both_acp_and_kokoro_failed",
                "_timestamp": time.time()
            }
    
    def _synthesize_with_acp(self, text: str, speaker_id: str, **kwargs) -> Dict[str, Any]:
        """ACP voice path - wrapped in exception handler"""
        return self.acp_voice.speak(text, speaker_id, **kwargs)
    
    def _is_valid_audio(self, result: Dict) -> bool:
        """Validate audio file exists and has content"""
        if not isinstance(result, dict):
            return False
        
        audio_path = result.get("audio_path")
        if not audio_path or not Path(audio_path).exists():
            return False
        
        # Check file size > 1KB (not empty)
        if Path(audio_path).stat().st_size < 1024:
            return False
        
        return True

    def get_health_status(self) -> Dict[str, Any]:
        """Monitoring endpoint"""
        baseline_status = self.kokoro.get_health_status()
        return {
            "acp_voice_available": self.acp_voice is not None,
            "kokoro_baseline_available": self.kokoro is not None,
            "kokoro_assets_configured": baseline_status["kokoro_assets_configured"],
            "edge_fallback_available": baseline_status["edge_available"],
            "training_mode": self.config.TRAINING_MODE,
            "fallback_was_used": self.fallback_was_used,
        }

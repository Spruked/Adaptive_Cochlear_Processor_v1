# Copyright (c) 2026 Adaptive Cochlear Processor Team
# Licensed under the MIT License. See LICENSE file for details.

"""
Main Orchestrator: Product-level integration of ACP + Fallbacks.
This is the only file your app imports.
"""
from router.stt_router import STTRouter
from router.tts_router import TTSRouter
from router.config import RouterConfig

class ACPHub:
    """
    Single API for your product:
    
    hub = ACPHub()
    transcript = hub.hear("audio.wav")
    audio_path = hub.speak("Hello world", speaker="phil")
    """
    
    def __init__(self):
        self.config = RouterConfig()
        self.stt = STTRouter()
        self.tts = TTSRouter()
        
        # Health check on startup
        self._perform_health_check()
    
    def hear(self, audio_path: str, context: dict = None) -> str:
        """
        Transcribe audio. Always succeeds.
        Returns: transcript string (source is transparent to caller)
        """
        result = self.stt.transcribe(audio_path, context)
        
        # Return just the text to maintain API simplicity
        # Full metadata available in result if needed
        return result["transcript"]
    
    def speak(self, text: str, speaker_id: str = None) -> str:
        """
        Synthesize speech. Always succeeds.
        Returns: path to audio file
        """
        result = self.tts.synthesize(text, speaker_id)
        return result["audio_path"]
    
    def _perform_health_check(self):
        """Log system state on startup"""
        import logging
        logging.info("🔍 ACP Hub Health Check:")
        logging.info(f"   ACP Enabled: {self.config.ACP_ENABLED}")
        logging.info(f"   Runtime: {'WSL2' if self.config.IS_WSL else 'Windows'}")
        logging.info(f"   STT Router: {self.stt.get_health_status()}")
        logging.info(f"   Training Mode: {self.config.TRAINING_MODE}")
    
    def get_system_status(self) -> dict:
        """Monitoring endpoint for dashboards"""
        return {
            "stt_status": self.stt.get_health_status(),
            "tts_status": self.tts.get_health_status(),
            "config": self.config.get_status(),
            "fallback_activated_stt": self.stt.fallback_was_used,
            "fallback_activated_tts": self.tts.fallback_was_used,
            "stt_metrics": self.stt.get_metrics()  # ← NEW
        }

# Usage in your app:
# from orchestrator import ACPHub
# hub = ACPHub()
# text = hub.hear("episode.wav")
# audio = hub.speak(text, speaker="phil_dandy")

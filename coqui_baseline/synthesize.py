# Copyright (c) 2026 Adaptive Cochlear Processor Team
# Licensed under the MIT License. See LICENSE file for details.

"""
Coqui Baseline: Stateless TTS, no SKG influence.
This lives on Windows running official coqui-tts.
"""
import hashlib  # ← NEW
from pathlib import Path
import os
from TTS.api import TTS
import time
from typing import Dict, Any

class CoquiBaseline:
    """
    Guaranteed voice synthesis. No adaptive parameters.
    """
    
    def __init__(self, model_name="tts_models/en/ljspeech/tacotron2-DDC"):
        self.tts = TTS(model_name=model_name, progress_bar=False)
    
    def synthesize(self, text: str, speaker_id: str = None) -> Dict[str, Any]:
        """
        Simple text-to-speech. Output goes directly to user.
        """
        # Portable cache directory
        base = Path(os.getenv("ACP_AUDIO_DIR", Path.cwd() / "audio_cache"))
        base.mkdir(exist_ok=True, parents=True)  # ← NEW: parents=True
        
        # Deterministic hash for repeatability
        key = hashlib.sha1(text.encode()).hexdigest()[:12]  # ← FIX #5 applied here
        output_path = base / f"coqui_{key}.wav"
        
        # Only synthesize if not cached (optimization)
        if not output_path.exists():
            self.tts.tts_to_file(text=text, file_path=str(output_path))
        
        return {
            "audio_path": str(output_path),  # ← NEW: explicit str()
            "speaker_id": speaker_id,
            "_source": "coqui_baseline",
            "_timestamp": time.time()
        }

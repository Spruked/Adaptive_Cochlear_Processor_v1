# Copyright (c) 2026 Adaptive Cochlear Processor Team
# Licensed under the MIT License. See LICENSE file for details.

"""
Whisper Baseline: Stateless, pristine, no learning.
This lives on Windows running the official whisper package.
"""
import whisper
import time
from typing import Dict, Any

class WhisperBaseline:
    """
    Guaranteed to work. No experimental features.
    """
    
    def __init__(self, model_name="base"):
        # Load on CPU (Windows server)
        self.model = whisper.load_model(model_name, device="cpu")
    
    def transcribe(self, audio_path: str) -> Dict[str, Any]:
        """
        Simple, reliable transcription.
        No confidence shaping, no perception, no learning.
        """
        result = self.model.transcribe(
            audio_path,
            language="en",
            fp16=False  # CPU-safe
        )
        
        return {
            "transcript": result["text"].strip(),
            "confidence": 0.8,  # Whisper doesn't give word-level, use heuristic
            "_source": "whisper_baseline",
            "_timestamp": time.time()
        }

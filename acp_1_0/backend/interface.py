# Copyright (c) 2026 Adaptive Cochlear Processor Team
# Licensed under the MIT License. See LICENSE file for details.

"""
Abstract backend interface: ACP doesn't care which decoder you use.
Swap WhisperTeacher → MinimalDecoder → CustomModel without changing perception.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any

class DecoderBackend(ABC):
    """
    All acoustic decoders must implement this.
    ACP only interacts with perception → DecoderBackend → cognition.
    """

    @abstractmethod
    def transcribe(self, audio_path: str) -> Dict[str, Any]:
        """
        Input: path to audio file
        Output: {
            "text": "full transcription",
            "words": [
                {"text": "the", "start": 0.0, "end": 0.3, "confidence": 0.9},
                ...
            ],
            "confidence": 0.75  # Overall
        }
        """
        pass

class WhisperDecoder(DecoderBackend):
    """Wrapper for initial testing (not shipped)"""
    def __init__(self):
        import whisper
        self.model = whisper.load_model("tiny", device="cpu")  # Tiny = fast

    def transcribe(self, audio_path):
        result = self.model.transcribe(
            audio_path,
            word_timestamps=True
        )

        words = []
        for segment in result["segments"]:
            if "words" in segment:
                words.extend(segment["words"])

        return {
            "text": result["text"],
            "words": words,
            "confidence": result.get("confidence", 0.8)
        }

class MinimalDecoderBackend(DecoderBackend):
    """Your eventual product decoder"""
    def __init__(self, model_path="models/minimal_decoder.pt"):
        from .minimal_decoder import MinimalDecoder
        self.decoder = MinimalDecoder(model_path)

    def transcribe(self, audio_path):
        # Load spectrogram
        import librosa
        audio, sr = librosa.load(audio_path, sr=16000)
        mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=80)

        return self.decoder.transcribe(mel)

# Copyright (c) 2026 Adaptive Cochlear Processor Team
# Licensed under the MIT License. See LICENSE file for details.

"""
AdaptiveCochlearProcessor (ACP) 1.0
Perception-first STT with intentional imperfection.
"""
import numpy as np
from .perception import PerceptualFilter
from .cognition import CognitiveEngine
from .learning import SKGLearning
from .backend.interface import DecoderBackend

class AdaptiveCochlearProcessor:
    """
    Single API: processor = ACP(); result = processor.hear('audio.wav')
    """

    def __init__(
        self,
        decoder: DecoderBackend = None,
        skg_path: str = "skg/default.json"
    ):
        # Perception layer (YOUR EDGE)
        self.perceptual = PerceptualFilter()

        # Cognition layer (error handling)
        self.cognition = CognitiveEngine()

        # Learning layer (memory)
        self.learning = SKGLearning(skg_path)

        # Acoustic decoder (pluggable backend)
        self.decoder = decoder or self._default_decoder()

        # State tracking
        self.session_corrections = []

    def _default_decoder(self):
        """Default to tiny Whisper for bootstrapping"""
        from .backend.interface import WhisperDecoder
        return WhisperDecoder()

    def hear(self, audio_path: str, context: dict = None) -> dict:
        """
        Main API: Simulates human hearing of audio file.

        Process:
        1. Load audio → perceptual filtering (makes it "harder")
        2. Decode → get imperfect transcript
        3. Cognition → detect mishearings & correct
        4. Learning → update mastery scores
        """

        # 1. Perceptual filtering (intentionally degrades audio)
        perceptual_audio, percept_report = self.perceptual.apply(
            audio_path, context
        )

        # 2. Decode (the "cochlea" part)
        # Backend can be swapped without changing perception
        decode_result = self.decoder.transcribe(perceptual_audio)

        # 3. Cognitive interpretation (human-like correction)
        # Cognition inspects confidence and decides what to "hear"
        interpreted = self.cognition.interpret(
            decode_result,
            percept_report,
            context
        )

        # 4. Learning from mistakes
        for correction in interpreted["corrections"]:
            phoneme = self._extract_phoneme(correction["original"])
            self.learning.update_mastery(
                phoneme=phoneme,
                was_misheard=True,
                context=context
            )

        # 5. Update perceptual filter with learning
        self.perceptual.adjust_from_learning(self.learning.get_state())

        # 6. Save persistent state
        self.learning.save()

        return {
            "final_transcript": interpreted["text"],
            "initial_transcript": decode_result["text"],
            "corrections": interpreted["corrections"],
            "confidence": interpreted["overall_confidence"],
            "mastery_updates": self.learning.get_state()["phoneme_mastery"]
        }

    def _extract_phoneme(self, word: str) -> str:
        """Map word to phoneme ID (simplified)"""
        # Use phonemizer library for production
        return word[:3].lower()

    def get_mastery_report(self) -> dict:
        """Export learning progress"""
        return self.learning.get_state()

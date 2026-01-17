# Copyright (c) 2026 Adaptive Cochlear Processor Team
# Licensed under the MIT License. See LICENSE file for details.

"""
CognitiveEngine: Models human interpretation of ambiguous audio.
"""
import numpy as np

class CognitiveEngine:
    """
    Inspects decoder output and perceptual state to:
    1. Detect likely mishearings (low confidence + context mismatch)
    2. Propose corrections
    3. Assign final confidence
    """

    def __init__(self, correction_threshold=0.6):
        self.threshold = correction_threshold

    def interpret(self, decode_result: dict, perceptual_report: dict, context: dict) -> dict:
        """
        Main cognition loop: decides what the "final heard version" is.
        """
        corrections = []
        final_words = []

        for word_data in decode_result.get("words", []):
            # Extract signal
            acoustic_confidence = word_data.get("confidence", 0.8)
            perceptual_confidence = perceptual_report["confidence_factor"]

            # Combined confidence
            confidence = acoustic_confidence * perceptual_confidence

            # Should we "hear" this differently?
            if confidence < self.threshold and self._context_suggests_correction(word_data, context):
                corrected = self._propose_correction(word_data, context)
                corrections.append({
                    "original": word_data["text"],
                    "corrected": corrected,
                    "confidence_before": confidence,
                    "reason": "low_confidence + context_mismatch"
                })
                final_words.append(corrected)
            else:
                final_words.append(word_data["text"])

        return {
            "text": " ".join(final_words),
            "corrections": corrections,
            "overall_confidence": np.mean([
                w.get("confidence", 0.8) for w in decode_result.get("words", [])
            ])
        }

    def _context_suggests_correction(self, word_data: dict, context: dict) -> bool:
        """Would a human doubt this word in this context?"""
        if not context:
            return False

        word = word_data["text"].lower()
        topic = context.get("topic", "")

        # Example: "aye" in AI context is suspicious
        if word == "aye" and "ai" in topic.lower():
            return True

        return False

    def _propose_correction(self, word_data: dict, context: dict) -> str:
        """Simple context-based correction"""
        # In production: use phoneme embeddings or small LM
        if word_data["text"].lower() == "aye":
            return "AI"
        return word_data["text"]

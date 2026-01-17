# Copyright (c) 2026 Adaptive Cochlear Processor Team
# Licensed under the MIT License. See LICENSE file for details.

"""
CognitiveEngine: Models human interpretation of ambiguous audio.
Inspects decoder output, detects mishearings, proposes corrections.
"""
import numpy as np
from typing import Dict, List, Any

class CognitiveEngine:
    """
    Decides what the "final heard version" is based on:
    - Acoustic confidence (from decoder)
    - Perceptual state (attention, dropouts)
    - Context (topic, speaker)

    Not a language model—it's a **confidence arbiter** with context hints.
    """

    def __init__(self, correction_threshold: float = 0.9):
        self.threshold = correction_threshold  # Below this: doubt the word
        self.context_window = []  # Last 10 words for local coherence

    def interpret(self, decode_result: Dict[str, Any],
                  perceptual_report: Dict[str, Any],
                  context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Main cognition loop. Returns final text + corrections made.
        """
        words = decode_result.get("words", [])
        if not words:
            # Fallback for non-word-level decoders
            words = [{"text": w, "confidence": 0.8}
                     for w in decode_result.get("text", "").split()]

        corrections = []
        final_words = []

        for i, word_data in enumerate(words):
            word = word_data.get("text", "")
            acoustic_conf = word_data.get("confidence", 0.8)
            perceptual_conf = perceptual_report.get("confidence_factor", 0.8)

            # Combined confidence (acoustic × perceptual)
            confidence = acoustic_conf * perceptual_conf

            # Should we doubt this word?
            if confidence < self.threshold:
                # Context check: is this word suspicious in this context?
                if self._is_suspicious(word, context, i):
                    corrected = self._propose_correction(word, context, i)
                    if corrected != word:
                        corrections.append({
                            "original": word,
                            "corrected": corrected,
                            "confidence_before": confidence,
                            "confidence_after": self._estimate_correction_confidence(corrected),
                            "position": i,
                            "reason": f"low_confidence ({confidence:.2f}) + context_mismatch"
                        })
                        final_words.append(corrected)
                        self._update_context(corrected)
                        continue

            # Default: accept the word
            final_words.append(word)
            self._update_context(word)

        return {
            "text": " ".join(final_words),
            "corrections": corrections,
            "overall_confidence": self._compute_overall_confidence(final_words, corrections),
            "context_snapshot": self.context_window[-5:]  # Last 5 words for learning
        }

    def _is_suspicious(self, word: str, context: Dict[str, Any], position: int) -> bool:
        """Would a human doubt this word in this context?"""
        word_lower = word.lower()

        # Keyword-triggered suspicion
        if context:
            topic = context.get("topic", "")

            # "aye" in AI context is suspicious
            if word_lower == "aye" and "ai" in topic.lower():
                return True

            # "loose" vs "lose" in tech context
            if word_lower == "loose" and any(w in topic for w in ["error", "bug", "fix"]):
                return True

            # "their/there/they're" confusion
            if word_lower in ["their", "there", "theyre"] and position < 3:
                return True

        # Phoneme-level suspicion: single-syllable words in noisy sections
        if len(word) <= 3 and self._recent_dropout():
            return True

        return False

    def _propose_correction(self, word: str, context: Dict[str, Any], position: int) -> str:
        """Simple context-aware correction (not a full LM)"""
        word_lower = word.lower()
        topic = context.get("topic", "") if context else ""

        # Direct mappings from SKG learning
        correction_map = {
            "aye": "AI",
            "loose": "lose",
            "theyre": "they're"
        }

        if word_lower in correction_map:
            return correction_map[word_lower]

        # Phonetic similarity (sounds-like)
        if word_lower == "plugin" and "ai" in topic:
            return "plug in"

        # Default: keep original
        return word

    def _estimate_correction_confidence(self, corrected_word: str) -> float:
        """Higher confidence for multi-character corrections"""
        base_confidence = 0.75
        length_bonus = min(len(corrected_word) * 0.02, 0.15)
        return min(1.0, base_confidence + length_bonus)

    def _compute_overall_confidence(self, words: List[str], corrections: List[Dict]) -> float:
        """Weighted average: corrections reduce overall confidence"""
        if not words:
            return 0.0

        # Start with perfect confidence
        total_confidence = 1.0

        # Deduct for each correction (more severe for early corrections)
        for corr in corrections:
            position_penalty = 1.0 - (corr["position"] / max(len(words), 1))
            confidence_penalty = (1.0 - corr["confidence_before"]) * position_penalty
            total_confidence -= confidence_penalty * 0.5

        return max(0.1, total_confidence)

    def _update_context(self, word: str):
        """Keep last 10 words for local coherence checks"""
        self.context_window.append(word)
        if len(self.context_window) > 10:
            self.context_window.pop(0)

    def _recent_dropout(self) -> bool:
        """Check if recent audio had perceptual dropouts"""
        # This would be passed from perceptual report in practice
        return np.random.random() < 0.3  # Simulated: 30% chance of recent dropout

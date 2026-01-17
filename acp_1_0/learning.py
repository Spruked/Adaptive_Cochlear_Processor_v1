# Copyright (c) 2026 Adaptive Cochlear Processor Team
# Licensed under the MIT License. See LICENSE file for details.

"""
SKGLearning: Persistent memory for phoneme and speaker mastery.
"""
import json
import os
from typing import Dict, Any

class SKGLearning:
    """
    Semantic Knowledge Graph (SKG) for learning from corrections.
    Persists mastery scores across sessions.
    """

    def __init__(self, skg_path: str = "skg.json"):
        self.skg_path = skg_path
        self.state = self._load_state()

    def _load_state(self) -> Dict[str, Any]:
        """Load SKG from disk"""
        if os.path.exists(self.skg_path):
            with open(self.skg_path, 'r') as f:
                return json.load(f)
        else:
            # Initialize empty SKG
            return {
                "phoneme_mastery": {},
                "speaker_profiles": {},
                "session_count": 0
            }

    def update_mastery(self, phoneme: str, was_misheard: bool, context: dict = None):
        """Update mastery score for a phoneme"""
        if phoneme not in self.state["phoneme_mastery"]:
            self.state["phoneme_mastery"][phoneme] = {
                "mastery_score": 0.5,  # Start neutral
                "total_encounters": 0,
                "mishear_count": 0
            }

        entry = self.state["phoneme_mastery"][phoneme]
        entry["total_encounters"] += 1

        if was_misheard:
            entry["mishear_count"] += 1

        # Simple mastery calculation: fewer mishears = higher mastery
        entry["mastery_score"] = 1.0 - (entry["mishear_count"] / entry["total_encounters"])

    def get_state(self) -> Dict[str, Any]:
        """Get current SKG state"""
        return self.state

    def save(self):
        """Persist SKG to disk"""
        with open(self.skg_path, 'w') as f:
            json.dump(self.state, f, indent=2)

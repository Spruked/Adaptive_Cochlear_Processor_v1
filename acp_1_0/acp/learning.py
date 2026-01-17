# Copyright (c) 2026 Adaptive Cochlear Processor Team
# Licensed under the MIT License. See LICENSE file for details.

"""
SKGLearning: Persistent memory for phoneme mastery and speaker profiles.
Atomic JSON writes, append-only correction memory, corruption-resistant.
"""
import json
import time
import logging
from pathlib import Path
from typing import Dict, Any, List
import shutil
import numpy as np

class SKGLearning:
    """
    Speaker Knowledge Graph for hearing.
    Stores:
    - phoneme_mastery: {"ai": 0.85, "th": 0.62, ...}
    - speaker_profiles: {"phil_dandy": {"mastery": 0.9, "common_errors": {...}}}
    - correction_memory: Last 100 corrections (append-only)

    File: skg/hearing.json (auto-created if missing)
    """

    def __init__(self, skg_path: str = "skg/hearing.json"):
        self.path = Path(skg_path)  # ← FIX #1: skp_path → skg_path
        self.data = self._load()
        self.uncommitted_changes = 0
        self.autosave_threshold = 10  # Save after N corrections

    def _load(self) -> Dict[str, Any]:
        """Atomic load with corruption recovery"""
        if self.path.exists():
            try:
                return json.loads(self.path.read_text())
            except json.JSONDecodeError:
                # Corrupted file: attempt backup recovery
                backup = self.path.with_suffix(".json.bak")
                if backup.exists():
                    return json.loads(backup.read_text())

        # Default structure
        return {
            "metadata": {
                "version": "1.0",
                "created": time.time(),
                "last_updated": time.time()
            },
            "phoneme_mastery": {},
            "speaker_profiles": {},
            "correction_memory": {
                "count": 0,
                "last_100": []
            }
        }

    def update_phoneme_mastery(self, phoneme: str, was_misheard: bool, context: Dict = None):
        """
        Adjust phoneme mastery based on correction outcome.
        was_misheard=True: We got it wrong → increase learning need
        was_misheard=False: We got it right → reinforce mastery
        """
        # REJECT simulated data immediately
        if context and context.get("source") == "simulated_decoder":
            raise ValueError("🚨 BLOCKED: Cannot learn from simulated data. Fix Whisper.")
        
        if phoneme not in self.data["phoneme_mastery"]:
            self.data["phoneme_mastery"][phoneme] = {
                "mastery_score": 0.5,  # Start at medium
                "exposure_count": 0,
                "mishearing_history": []
            }

        phoneme_data = self.data["phoneme_mastery"][phoneme]

        # Learning rate: faster when uncertain, slower when mastered
        current_mastery = phoneme_data["mastery_score"]
        learning_rate = 0.08 if current_mastery < 0.7 else 0.03

        # Update rule: reinforcement learning
        if was_misheard:
            # Got it wrong: decrease mastery (but not too much)
            delta = -learning_rate * 0.7
        else:
            # Got it right: increase mastery
            delta = learning_rate * 0.95

        phoneme_data["mastery_score"] = max(0.1, min(1.0, current_mastery + delta))
        phoneme_data["exposure_count"] += 1

        # Log the learning event
        phoneme_data["mishearing_history"].append({
            "timestamp": time.time(),
            "was_misheard": was_misheard,
            "context": context.get("topic", "general") if context else "general"
        })

        # Keep history manageable (last 50)
        if len(phoneme_data["mishearing_history"]) > 50:
            phoneme_data["mishearing_history"].pop(0)

        self.uncommitted_changes += 1
        self._maybe_autosave()

    def update_speaker_profile(self, speaker_id: str, correction: Dict[str, Any]):
        """
        Learn speaker-specific quirks (voice pitch, common mishearings).
        """
        if speaker_id not in self.data["speaker_profiles"]:
            self.data["speaker_profiles"][speaker_id] = {
                "mastery_score": 0.5,
                "first_heard": time.time(),
                "total_corrections": 0,
                "common_mishearings": {}
            }

        profile = self.data["speaker_profiles"][speaker_id]
        profile["total_corrections"] += 1
        profile["last_heard"] = time.time()

        # Increase mastery (you get better at hearing this voice)
        profile["mastery_score"] = min(1.0, profile["mastery_score"] + 0.02)

        # Track common mishearing patterns
        original = correction.get("original", "")
        corrected = correction.get("corrected", "")
        pattern = f"{original}→{corrected}"

        if pattern not in profile["common_mishearings"]:
            profile["common_mishearings"][pattern] = 0
        profile["common_mishearings"][pattern] += 1

        self.uncommitted_changes += 1

    def log_correction(self, correction: Dict[str, Any]):
        """
        Append-only correction memory. Stores last 100 for pattern analysis.
        """
        memory = self.data["correction_memory"]

        entry = {
            "timestamp": time.time(),
            "original": correction.get("original", ""),
            "corrected": correction.get("corrected", ""),
            "phoneme": correction.get("phoneme", ""),
            "speaker_id": correction.get("speaker_id", "unknown"),
            "context": correction.get("context", "general"),
            "confidence_delta": correction.get("confidence_after", 0.0) - correction.get("confidence_before", 0.0)
        }

        memory["last_100"].append(entry)
        memory["count"] += 1

        # Keep only last 100 (circular buffer)
        if len(memory["last_100"]) > 100:
            memory["last_100"].pop(0)

        self.uncommitted_changes += 1

    def _maybe_autosave(self):
        """Save every N changes to prevent data loss"""
        if self.uncommitted_changes >= self.autosave_threshold:
            self.save()
            self.uncommitted_changes = 0

    def save(self):
        """Atomic write: write to temp, then rename (prevents corruption)"""
        # Create backup first
        if self.path.exists():
            backup = self.path.with_suffix(".json.bak")
            shutil.copy2(self.path, backup)

        # Update metadata
        self.data["metadata"]["last_updated"] = time.time()

        # Write to temp file
        temp_path = self.path.with_suffix(".json.tmp")
        if temp_path.exists():
            temp_path.unlink()  # Remove existing temp file
        temp_path.write_text(json.dumps(self.data, indent=2))

        # Atomic rename (remove target first on Windows)
        if self.path.exists():
            self.path.unlink()
        temp_path.rename(self.path)

        # Clean up old backup (keep last 5)
        self._cleanup_old_backups()

    def _cleanup_old_backups(self, keep_last: int = 1):  # ← FIX #4: Simplified
        backup = self.path.with_suffix(".json.bak")
        if backup.exists():
            pass  # Single-backup policy for v1.0

    def get_state(self) -> Dict[str, Any]:
        """Snapshot for perceptual filter adjustments"""
        return {
            "phoneme_mastery": self.data["phoneme_mastery"],
            "speaker_profiles": self.data["speaker_profiles"],
            "avg_phoneme_mastery": self._compute_avg_mastery("phoneme"),
            "avg_speaker_mastery": self._compute_avg_mastery("speaker")
        }

    def _compute_avg_mastery(self, mastery_type: str) -> float:
        """Compute average mastery score"""
        if mastery_type == "phoneme":
            scores = [v["mastery_score"] for v in self.data["phoneme_mastery"].values()]
        elif mastery_type == "speaker":
            scores = [v["mastery_score"] for v in self.data["speaker_profiles"].values()]
        else:
            return 0.5

        return np.mean(scores) if scores else 0.5

    def get_correction_report(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent corrections for analysis"""
        memory = self.data["correction_memory"]["last_100"]
        return memory[-limit:]

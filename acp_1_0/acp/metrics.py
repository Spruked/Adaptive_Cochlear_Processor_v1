# Copyright (c) 2026 Adaptive Cochlear Processor Team
# Licensed under the MIT License. See LICENSE file for details.

"""
MetricsTracker: Quantify learning gains with SKG deltas.
Logs JSON snapshots before/after each transcription.
No visualizations, no external deps, just data.
"""
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np

class MetricsTracker:
    """
    Tracks ACP learning progress via SKG state deltas.
    Logs: mastery trends, correction rates, confidence EMA.
    """

    def __init__(self, log_path: str = "acp_metrics.jsonl"):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(exist_ok=True)

        # Rolling state for EMA calculation
        self.last_snapshot = None
        self.confidence_ema = None
        self.ema_alpha = 0.3  # EMA smoothing factor

        # Running totals for corrections_per_100_words
        self.total_words = 0
        self.total_corrections = 0

    def record_pre(self, skg_state: Dict[str, Any]):
        """Capture SKG state *before* hear() call"""
        self.last_snapshot = {
            "timestamp": time.time(),
            "phoneme_mastery": skg_state.get("avg_phoneme_mastery", 0.5),
            "speaker_mastery": skg_state.get("avg_speaker_mastery", 0.5),
            "total_words": self.total_words,
            "total_corrections": self.total_corrections
        }

    def record_post(self, result: Dict[str, Any], skg_state: Dict[str, Any]):
        """
        Capture state *after* hear() call, compute deltas, log JSON.
        Returns metrics dict for immediate inspection.
        """
        if self.last_snapshot is None:
            return {}  # No pre-record, can't compute deltas

        # Count words in this transcription
        transcript = result.get("transcript", "")
        words_in_run = len(transcript.split())
        corrections_in_run = len(result.get("corrections", []))

        # Update running totals
        self.total_words += words_in_run
        self.total_corrections += corrections_in_run

        # Compute metrics
        current_confidence = result.get("confidence", 0.5)

        # EMA of confidence
        if self.confidence_ema is None:
            self.confidence_ema = current_confidence
        else:
            self.confidence_ema = (self.ema_alpha * current_confidence +
                                 (1 - self.ema_alpha) * self.confidence_ema)

        # Corrections per 100 words (rolling rate)
        corrections_per_100 = (self.total_corrections / self.total_words * 100) if self.total_words > 0 else 0

        # Build metrics payload
        metrics = {
            "timestamp": time.time(),
            "session_id": result.get("_session_id", "unknown"),

            # Raw values
            "phoneme_mastery": skg_state.get("avg_phoneme_mastery", 0.5),
            "speaker_mastery": skg_state.get("avg_speaker_mastery", 0.5),

            # Deltas (change from pre to post)
            "phoneme_mastery_delta": skg_state.get("avg_phoneme_mastery", 0.5) - self.last_snapshot["phoneme_mastery"],
            "speaker_mastery_delta": skg_state.get("avg_speaker_mastery", 0.5) - self.last_snapshot["speaker_mastery"],

            # Per-transcription stats
            "confidence": current_confidence,
            "confidence_ema": self.confidence_ema,
            "words_in_run": words_in_run,
            "corrections_in_run": corrections_in_run,

            # Rolling aggregates
            "total_words": self.total_words,
            "total_corrections": self.total_corrections,
            "corrections_per_100_words": corrections_per_100,

            # Learning signal (critical metric)
            "learning_signal": int(skg_state.get("avg_phoneme_mastery", 0.5) > self.last_snapshot["phoneme_mastery"] or
                                  skg_state.get("avg_speaker_mastery", 0.5) > self.last_snapshot["speaker_mastery"])
        }

        # Append to JSONL log (one line per transcription)
        with self.log_path.open("a") as f:
            f.write(json.dumps(metrics) + "\n")

        # Reset pre-snapshot
        self.last_snapshot = None

        return metrics

    def get_summary(self, last_n: int = 10) -> Dict[str, Any]:
        """
        Get summary of last N transcriptions.
        No external tools needed - just read the JSONL file.
        """
        if not self.log_path.exists():
            return {"error": "No metrics logged yet"}

        # Read last N lines
        lines = self.log_path.read_text().strip().split("\n")
        recent = [json.loads(line) for line in lines[-last_n:] if line.strip()]

        if not recent:
            return {"error": "No valid metrics"}

        # Compute trends
        recent_confidences = [m["confidence"] for m in recent]
        recent_corrections_per_100 = [m["corrections_per_100_words"] for m in recent]

        return {
            "total_transcriptions": len(lines),
            "last_n": last_n,
            "confidence_trend": {
                "start": recent_confidences[0],
                "end": recent_confidences[-1],
                "delta": recent_confidences[-1] - recent_confidences[0],
                "ema": recent[-1]["confidence_ema"]
            },
            "corrections_trend": {
                "start": recent_corrections_per_100[0],
                "end": recent_corrections_per_100[-1],
                "delta": recent_corrections_per_100[-1] - recent_corrections_per_100[0]
            },
            "learning_active": sum(m["learning_signal"] for m in recent) > (last_n / 2),
            "latest_mastery": {
                "phoneme": recent[-1]["phoneme_mastery"],
                "speaker": recent[-1]["speaker_mastery"]
            }
        }

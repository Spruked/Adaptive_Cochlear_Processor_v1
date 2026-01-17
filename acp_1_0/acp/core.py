# Copyright (c) 2026 Adaptive Cochlear Processor Team
# Licensed under the MIT License. See LICENSE file for details.

"""
AdaptiveCochlearProcessor (ACP) 1.0 - Core Orchestrator
Single API: processor = ACP(); result = processor.hear('audio.wav')
Wires together perception → decode → cognition → learning
"""
import time
from pathlib import Path
from typing import Dict, Any, Optional

# ACP internal layers
from .perception import PerceptualFilter
from .cognition import CognitiveEngine
from .learning import SKGLearning
from .metrics import MetricsTracker

# Pluggable decoder backend
from .backend.interface import DecoderBackend
from .backend.interface import WhisperDecoder  # Default for bootstrapping

class AdaptiveCochlearProcessor:
    """
    Main entry point for ACP 1.0.
    Router calls this. This calls perception, decoder, cognition, learning.
    """
    
    def __init__(self, 
                 skg_path: str = "skg/hearing.json",
                 decoder: Optional[DecoderBackend] = None):
        """
        Initialize all layers:
        - PerceptualFilter: human-like hearing simulation
        - DecoderBackend: acoustic decoder (Whisper, MinimalDecoder, etc)
        - CognitiveEngine: error detection & correction
        - SKGLearning: persistent memory
        """
        # Perception layer (your unique edge)
        self.perceptual = PerceptualFilter(sample_rate=16000)
        
        # Decoder (pluggable, starts with Whisper)
        self.decoder = decoder or WhisperDecoder(model_name="tiny")
        
        # Cognition layer (confidence arbiter)
        self.cognition = CognitiveEngine(correction_threshold=0.6)
        
        # Learning layer (persistent SKG)
        self.learning = SKGLearning(skg_path=skg_path)
        
        # Metrics tracking (quantify learning progress)
        self.metrics = MetricsTracker(log_path="acp_metrics_live.jsonl")  # ← FIXED: live metrics
        
        # Session tracking
        self.session_id = f"session_{int(time.time())}"
        self.processing_count = 0
    
    def hear(self, audio_path: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Single method API: Simulates human hearing of audio file.
        
        Flow:
        1. Perceptual filtering (degrades audio intentionally)
        2. Acoustic decoding (gets imperfect transcript)
        3. Cognitive interpretation (detects & corrects mishearings)
        4. Learning update (saves corrections to SKG)
        5. Return final result for router consumption
        
        Returns:
        {
            "transcript": "final corrected text",
            "confidence": 0.72,
            "corrections": [...],
            "perceptual_report": {...},
            "learning_snapshot": {...}
        }
        """
        start_time = time.time()
        self.processing_count += 1
        
        # Record pre-state for metrics
        skg_pre = self.learning.get_state()
        self.metrics.record_pre(skg_pre)
        
        # 1. Perceptual filtering (makes hearing "harder")
        # Applies frequency masking, attention fatigue, simulated dropouts
        perceptual_audio_path, perceptual_report = self.perceptual.apply(
            audio_path, 
            context
        )
        
        # 2. Acoustic decoding (gets initial transcript)
        # Can be Whisper, MinimalDecoder, or any backend implementing DecoderBackend
        decode_result = self.decoder.transcribe(perceptual_audio_path)
        
        # Check if this was simulated decoding
        if decode_result.get("_backend") == "whisper_simulated" or decode_result.get("_backend") == "whisper_simulated_variable":
            # Mark context as simulated for learning guardrail
            if context is None:
                context = {}
            context["source"] = "simulated_decoder"
        
        # 3. Cognitive interpretation (decides what was *actually* heard)
        # Inspects confidence, context, perceptual state to propose corrections
        cognitive_result = self.cognition.interpret(
            decode_result,
            perceptual_report,
            context
        )
        
        # 4. Learning from mistakes (updates SKG)
        # For each correction: update phoneme mastery + speaker profile
        for correction in cognitive_result["corrections"]:
            phoneme = self._extract_phoneme(correction["original"])
            
            # Update phoneme mastery (we misheard this)
            self.learning.update_phoneme_mastery(
                phoneme=phoneme,
                was_misheard=True,
                context=context
            )
            
            # Update speaker profile (if speaker known)
            if context and context.get("speaker_id"):
                self.learning.update_speaker_profile(
                    speaker_id=context["speaker_id"],
                    correction=correction
                )
            
            # Log correction for pattern analysis
            correction_with_meta = {
                **correction,
                "phoneme": phoneme,
                "speaker_id": context.get("speaker_id", "unknown"),
                "context": context.get("topic", "general") if context else "general"
            }
            self.learning.log_correction(correction_with_meta)
        
        # Also reinforce phonemes we got *right* (no corrections)
        for word in cognitive_result["text"].split():
            phoneme = self._extract_phoneme(word)
            # Check if this phoneme was in the original decode but not corrected
            was_corrected = any(
                correction["corrected"] == word 
                for correction in cognitive_result["corrections"]
            )
            if not was_corrected:
                self.learning.update_phoneme_mastery(
                    phoneme=phoneme,
                    was_misheard=False,
                    context=context
                )
        
        # 5. Adjust perceptual filter based on learning
        # If we're mastering certain phonemes, tune ear sensitivity
        self.perceptual.adjust_from_learning(self.learning.get_state())
        
        # 6. Atomic save of SKG (every 10 changes)
        self.learning.save()
        
        # 7. Build result dict first (needed for metrics)
        result = {
            "transcript": cognitive_result["text"],  # Expected by router
            "confidence": cognitive_result["overall_confidence"],
            "corrections": cognitive_result["corrections"],
            "perceptual_report": perceptual_report,
            "learning_snapshot": self.learning.get_state(),
            "_source": "acp",
            "_processing_time_ms": (time.time() - start_time) * 1000
        }
        
        # 8. Record metrics (quantify learning progress)
        skg_post = self.learning.get_state()
        metrics = self.metrics.record_post(result, skg_post)
        result["_metrics"] = metrics  # Add metrics to result
        
        return result
    
    def _extract_phoneme(self, word: str) -> str:
        """
        Map word to phoneme ID (simplified for v1.0).
        Production: use phonemizer library for proper phoneme extraction.
        """
        # Take first 2-3 letters + vowel count as simple proxy
        base = word.lower()[:3]
        vowels = sum(1 for c in word if c in 'aeiou')
        return f"{base}_v{vowels}"
    
    def get_mastery_report(self) -> Dict[str, Any]:
        """
        Export learning progress for monitoring/dashboards.
        """
        return self.learning.get_state()
    
    def get_correction_history(self, limit: int = 10) -> list:
        """
        Recent corrections for debugging/analysis.
        """
        return self.learning.get_correction_report(limit)

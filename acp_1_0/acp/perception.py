# Copyright (c) 2026 Adaptive Cochlear Processor Team
# Licensed under the MIT License. See LICENSE file for details.

"""
PerceptualFilter: Human-like audio processing
Simulates ear limitations, attention fatigue, and mishearing.
"""
import librosa
import numpy as np
from scipy.signal import butter, lfilter
from pathlib import Path
from typing import Dict, Any

class PerceptualFilter:
    def __init__(self, sample_rate=16000):
        self.sr = sample_rate
        self.attention_level = 0.8  # 0.0=exhausted, 1.0=hyper-focused
        
        # Human ear sensitivity: inverted U-curve (2-5kHz peak)
        self.freq_sensitivity = self._init_sensitivity_curve()
        
    def _init_sensitivity_curve(self):
        """20Hz-20kHz human hearing curve"""
        return {
            20: 0.10, 100: 0.30, 500: 0.60, 2000: 1.0,
            5000: 1.0, 10000: 0.70, 15000: 0.40, 20000: 0.15
        }
    
    def apply(self, audio_path: str, context: dict = None) -> tuple:
        """Process audio through human hearing simulation"""
        audio, _ = librosa.load(audio_path, sr=self.sr)
        
        # 1. Frequency masking (ear's natural filter)
        filtered = self._frequency_masking(audio)
        
        # 2. Attention gating (fatigue reduces amplitude)
        filtered = self._attention_gate(filtered)
        
        # 3. Temporal smearing (neural processing lag)
        filtered = self._temporal_smearing(filtered)
        
        # 4. Simulated dropouts (random mishearing)
        filtered, dropouts = self._simulated_dropout(filtered)
        
        # Save processed audio
        output_path = str(Path(audio_path).with_suffix(".perceptual.wav"))
        import soundfile as sf
        sf.write(output_path, filtered, self.sr)
        
        return output_path, {
            "attention_level": self.attention_level,
            "dropout_count": len(dropouts),
            "confidence_factor": self.attention_level * 0.95
        }
    
    def _frequency_masking(self, audio: np.ndarray) -> np.ndarray:
        """FFT-based frequency filtering (real DSP)"""
        fft = np.fft.rfft(audio)
        freqs = np.fft.rfftfreq(len(audio), 1/self.sr)
        
        for i, f in enumerate(freqs):
            sensitivity = self._interpolate_sensitivity(f)
            fft[i] *= sensitivity
        
        return np.fft.irfft(fft)
    
    def _interpolate_sensitivity(self, freq: float) -> float:
        """Linear interpolation from sensitivity curve"""
        freqs = sorted(self.freq_sensitivity.keys())
        for i in range(len(freqs) - 1):
            if freqs[i] <= freq <= freqs[i + 1]:
                low, high = freqs[i], freqs[i + 1]
                ratio = (freq - low) / (high - low)
                return self.freq_sensitivity[low] + ratio * (self.freq_sensitivity[high] - self.freq_sensitivity[low])
        return 0.1
    
    def _attention_gate(self, audio: np.ndarray) -> np.ndarray:
        """Attention acts as amplitude gate (0.7 to 1.0x)"""
        return audio * (0.7 + self.attention_level * 0.3)
    
    def _temporal_smearing(self, audio: np.ndarray) -> np.ndarray:
        """Blurs rapid transients when fatigued (<0.5 attention)"""
        if self.attention_level < 0.5:
            b, a = butter(2, 0.5)  # Low-pass filter
            return lfilter(b, a, audio)
        return audio
    
    def _simulated_dropout(self, audio: np.ndarray, rate: float = 0.02) -> tuple:
        """Random 20ms silences (models mishearing)"""
        dropout_len = int(0.02 * self.sr)
        n_dropouts = int(len(audio) * rate / dropout_len)
        
        dropouts = []
        for _ in range(n_dropouts):
            start = np.random.randint(0, len(audio) - dropout_len)
            audio[start:start + dropout_len] = 0
            dropouts.append({"start_ms": start / self.sr * 1000, "duration_ms": 20})
        
        return audio, dropouts
    
    def adjust_from_learning(self, skg_state: Dict[str, Any]):
        """
        Learning boosts attention for mastered speakers/phonemes.
        Called after each transcription to update perceptual state.
        """
        avg_phoneme_mastery = skg_state.get("avg_phoneme_mastery", 0.5)
        avg_speaker_mastery = skg_state.get("avg_speaker_mastery", 0.5)
        
        # Higher mastery → higher attention (you're better at hearing this)
        self.attention_level = min(1.0, 0.6 + (avg_phoneme_mastery * 0.4))
        
        # Boost frequency sensitivity in mastered ranges
        if avg_speaker_mastery > 0.8:
            for f in self.freq_sensitivity:
                if 2000 <= f <= 5000:
                    self.freq_sensitivity[f] = min(1.0, self.freq_sensitivity[f] * 1.1)

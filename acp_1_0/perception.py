# Copyright (c) 2026 Adaptive Cochlear Processor Team
# Licensed under the MIT License. See LICENSE file for details.

"""
PerceptualFilter: models human hearing limitations.
This is where ACP's unique value lives.
"""
import librosa
import numpy as np
from scipy.signal import butter, lfilter

class PerceptualFilter:
    """
    Human-like filtering:
    - Frequency masking (ear sensitivity curve)
    - Attention gating (fatigue)
    - Temporal smearing (neural lag)
    - Simulated dropouts (mishearing)
    """

    def __init__(self, sample_rate=16000):
        self.sr = sample_rate
        self.attention_level = 0.8  # 0.0 = exhausted, 1.0 = hyper-focused

        # Frequency sensitivity: inverted U-curve (2-5kHz peak)
        self.freq_sensitivity = self._init_sensitivity_curve()

    def apply(self, audio_path: str, context: dict = None) -> tuple:
        """
        Process audio through human hearing simulation.
        Returns: (filtered_audio_path, perceptual_report)
        """
        # Load
        audio, _ = librosa.load(audio_path, sr=self.sr)

        # Apply perceptual effects
        audio = self._frequency_masking(audio)
        audio = self._attention_gate(audio)
        audio = self._temporal_smearing(audio)
        audio, dropouts = self._simulated_dropout(audio)

        # Save filtered version
        import soundfile as sf
        output_path = audio_path.replace(".wav", "_perceptual.wav")
        sf.write(output_path, audio, self.sr)

        report = {
            "attention_level": self.attention_level,
            "dropout_count": len(dropouts),
            "confidence_factor": self.attention_level * 0.95
        }

        return output_path, report

    def _frequency_masking(self, audio: np.ndarray) -> np.ndarray:
        """FFT-based frequency filtering"""
        fft = np.fft.rfft(audio)
        freqs = np.fft.rfftfreq(len(audio), 1/self.sr)

        for i, f in enumerate(freqs):
            # Human ear curve
            if 2000 <= f <= 5000:
                fft[i] *= 1.15  # Boost speech frequencies
            elif f < 100 or f > 15000:
                fft[i] *= 0.3  # Attenuate extremes

        return np.fft.irfft(fft)

    def _attention_gate(self, audio: np.ndarray) -> np.ndarray:
        """Attention acts as amplitude gate"""
        gate_factor = 0.7 + (self.attention_level * 0.3)
        return audio * gate_factor

    def _temporal_smearing(self, audio: np.ndarray) -> np.ndarray:
        """Blurs rapid transients when fatigued"""
        if self.attention_level < 0.5:
            # Low-pass filter
            b, a = butter(2, 0.5)  # 0.5 * Nyquist
            return lfilter(b, a, audio)
        return audio

    def _simulated_dropout(self, audio: np.ndarray, rate: float = 0.02):
        """Random 20ms segments of silence (mishearing)"""
        dropout_len = int(0.02 * self.sr)
        n_dropouts = int(len(audio) * rate / dropout_len)

        dropouts = []
        for _ in range(n_dropouts):
            start = np.random.randint(0, len(audio) - dropout_len)
            audio[start:start+dropout_len] = 0
            dropouts.append({"start_sample": start, "duration_ms": 20})

        return audio, dropouts

    def adjust_from_learning(self, skg_state: dict):
        """Learning boosts attention for mastered speakers"""
        avg_mastery = np.mean([
            v["mastery_score"]
            for v in skg_state.get("phoneme_mastery", {}).values()
        ] or [0.5])

        # Better mastery → higher attention
        self.attention_level = min(1.0, 0.7 + (avg_mastery * 0.3))

    def _init_sensitivity_curve(self):
        """Initialize frequency sensitivity curve"""
        # Placeholder for more complex curve
        return np.ones(80)  # For mel bins

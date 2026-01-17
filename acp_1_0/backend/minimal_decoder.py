# Copyright (c) 2026 Adaptive Cochlear Processor Team
# Licensed under the MIT License. See LICENSE file for details.

"""
MinimalDecoder: Small acoustic model that runs fast on CPU.
Initially uses Whisper's *architecture* but tiny size.
"""
import numpy as np
import torch

class MinimalDecoder:
    """
    ACP's own acoustic decoder.
    - Starts as a tiny Conv-RNN
    - Input: log-mel spectrogram (extracted by teacher)
    - Output: phoneme sequence + confidence scores
    """

    def __init__(self, model_path="models/minimal_decoder.pt"):
        # Tiny model: 1 conv layer + 1 LSTM = <10MB
        self.model = self._load_model(model_path)
        self.model.eval()  # No training during inference

    def transcribe(self, spectrogram):
        """
        Input: mel spectrogram [80, 3000] (timesteps)
        Output: {'text': ..., 'confidence': [...]}
        """
        with torch.no_grad():
            # Simple forward pass
            features = self.model.encoder(spectrogram)
            tokens = self.model.decoder(features)

            text = self._detokenize(tokens)
            confidence = self._estimate_confidence(features)

        return {
            "text": text,
            "confidence": confidence,
            "was_corrected": False  # Flag for learning loop
        }

    def _load_model(self, path):
        """Load your trained tiny model"""
        # Initially: load a dummy if not trained
        try:
            return torch.load(path, map_location="cpu")
        except FileNotFoundError:
            print("⚠️ No trained decoder found. Using dummy (random).")
            return self._dummy_model()

    def _dummy_model(self):
        """Fallback: random decoder for testing perception layer"""
        class DummyModel:
            class Encoder:
                def __call__(self, x):
                    return torch.randn(128, 100)  # Fake features

            class Decoder:
                def __call__(self, features):
                    return "dummy transcription"

            encoder = Encoder()
            decoder = Decoder()

        return DummyModel()

    def _detokenize(self, tokens):
        """Convert token IDs to text"""
        # Use simple greedy decoding
        # In production: beam search with phoneme constraints
        return tokens

    def _estimate_confidence(self, features):
        """From feature variance (proxy for uncertainty)"""
        # High variance = noisy = low confidence
        variance = torch.var(features).item()
        return max(0.1, min(1.0, 1.0 / (1.0 + variance)))

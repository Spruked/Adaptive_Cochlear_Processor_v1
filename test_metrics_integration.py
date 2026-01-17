# Copyright (c) 2026 Adaptive Cochlear Processor Team
# Licensed under the MIT License. See LICENSE file for details.

#!/usr/bin/env python3
"""
Test metrics integration with ACP core
"""
import numpy as np
import soundfile as sf
from acp_1_0.acp.core import AdaptiveCochlearProcessor

# Create test audio file
sample_rate = 16000
duration = 1.0
frequency = 440.0
t = np.linspace(0, duration, int(sample_rate * duration), False)
audio = np.sin(frequency * 2 * np.pi * t)
sf.write('test_audio.wav', audio, sample_rate)

# Initialize ACP with metrics
processor = AdaptiveCochlearProcessor(skg_path='test_skg.json')

# Process audio
result = processor.hear('test_audio.wav', context={'topic': 'AI', 'speaker_id': 'test_speaker'})

# Display results
print('Transcript:', result['transcript'])
print('Confidence:', f"{result['confidence']:.2f}")
print('Corrections:', len(result.get('corrections', [])))

# Display metrics
metrics = result.get('_metrics', {})
if metrics:
    print('\n📊 Metrics:')
    print('Learning Signal:', metrics.get('learning_signal', False))
    print('Corrections/100w:', f"{metrics.get('corrections_per_100_words', 0):.1f}")
    print('Phoneme Mastery:', f"{metrics.get('phoneme_mastery', 0.5):.3f}")
    print('Confidence EMA:', f"{metrics.get('confidence_ema', 0.5):.3f}")
else:
    print('No metrics recorded')

# Cleanup
import os
os.remove('test_audio.wav')
os.remove('test_skg.json')  # Remove test SKG

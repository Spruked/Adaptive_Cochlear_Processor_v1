# Copyright (c) 2026 Adaptive Cochlear Processor Team
# Licensed under the MIT License. See LICENSE file for details.

from setuptools import setup, find_packages

setup(
    name="acp-1.0",
    version="1.0.0",
    description="Adaptive Cochlear Processor - Human-like STT",
    packages=find_packages(),
    install_requires=[
        "librosa>=0.10.0",
        "scipy>=1.10.0",
        "numpy>=1.24.0",
        "soundfile>=0.12.0",
        "torch>=2.0.0",  # For decoder
        "phonemizer>=3.2.0",  # For phoneme extraction
        "edge-tts>=7.2.3",
        "kokoro-onnx>=0.5.0",
    ],
    extras_require={
        "dev": ["pytest", "black", "whisper"],  # Whisper only for teaching
    },
    python_requires=">=3.8",
    author="Your Name",
    url="https://github.com/yourusername/acp-1.0",
)

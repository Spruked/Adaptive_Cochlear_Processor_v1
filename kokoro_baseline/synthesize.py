# Copyright (c) 2026 Adaptive Cochlear Processor Team
# Licensed under the MIT License. See LICENSE file for details.

"""
Kokoro Baseline: Deterministic TTS with Edge fallback.
This keeps the baseline lightweight while preserving the "always succeeds" contract.
"""
from __future__ import annotations

import asyncio
import hashlib
import os
import wave
from pathlib import Path
from typing import Any, Dict

try:
    import edge_tts
except ImportError:  # pragma: no cover - runtime dependency may be absent
    edge_tts = None

try:
    import numpy as np
    from kokoro_onnx import Kokoro
except ImportError:  # pragma: no cover - runtime dependency may be absent
    np = None
    Kokoro = None


class KokoroBaseline:
    """
    Guaranteed voice synthesis with Kokoro as the primary baseline.
    If Kokoro assets are not configured, Edge TTS preserves the safety net.
    """

    def __init__(self):
        self.audio_dir = Path(os.getenv("ACP_AUDIO_DIR", Path.cwd() / "audio_cache"))
        self.audio_dir.mkdir(exist_ok=True, parents=True)
        self.kokoro_model_path = os.getenv("ACP_KOKORO_MODEL_PATH", "").strip()
        self.kokoro_voices_path = os.getenv("ACP_KOKORO_VOICES_PATH", "").strip()
        self.kokoro_voice = os.getenv("ACP_KOKORO_DEFAULT_VOICE", "af_bella").strip()
        self.kokoro_lang = os.getenv("ACP_KOKORO_DEFAULT_LANG", "en-us").strip()
        self.edge_voice = os.getenv("ACP_EDGE_DEFAULT_VOICE", "en-US-EmmaMultilingualNeural").strip()

    def synthesize(self, text: str, speaker_id: str = None) -> Dict[str, Any]:
        key = hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]

        if self._kokoro_ready():
            output_path = self.audio_dir / f"kokoro_{key}.wav"
            if not output_path.exists():
                self._synthesize_with_kokoro(text, output_path)
            return {
                "audio_path": str(output_path),
                "speaker_id": speaker_id,
                "_source": "kokoro_baseline",
                "_provider": "kokoro",
            }

        if self._edge_ready():
            output_path = self.audio_dir / f"edge_{key}.mp3"
            if not output_path.exists():
                self._synthesize_with_edge(text, output_path)
            return {
                "audio_path": str(output_path),
                "speaker_id": speaker_id,
                "_source": "kokoro_baseline",
                "_provider": "edge_tts",
                "_fallback_reason": "kokoro_assets_not_configured",
            }

        raise RuntimeError("Neither Kokoro nor Edge TTS is available.")

    def _kokoro_ready(self) -> bool:
        return (
            Kokoro is not None
            and bool(self.kokoro_model_path)
            and bool(self.kokoro_voices_path)
            and Path(self.kokoro_model_path).exists()
            and Path(self.kokoro_voices_path).exists()
        )

    def _edge_ready(self) -> bool:
        return edge_tts is not None

    def get_health_status(self) -> Dict[str, Any]:
        return {
            "kokoro_assets_configured": self._kokoro_ready(),
            "edge_available": self._edge_ready(),
            "audio_dir": str(self.audio_dir),
        }

    def _synthesize_with_kokoro(self, text: str, output_path: Path) -> None:
        kokoro = Kokoro(self.kokoro_model_path, self.kokoro_voices_path)
        audio, sample_rate = kokoro.create(text, voice=self.kokoro_voice, lang=self.kokoro_lang)
        self._write_wav(output_path, audio, sample_rate)

    async def _synthesize_with_edge_async(self, text: str, output_path: Path) -> None:
        communicator = edge_tts.Communicate(text=text, voice=self.edge_voice)
        await communicator.save(output_path)

    def _synthesize_with_edge(self, text: str, output_path: Path) -> None:
        asyncio.run(self._synthesize_with_edge_async(text, output_path))

    def _write_wav(self, output_path: Path, audio: "np.ndarray", sample_rate: int) -> None:
        pcm_audio = np.clip(audio, -1.0, 1.0)
        pcm_audio = (pcm_audio * 32767).astype(np.int16)
        with wave.open(str(output_path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(pcm_audio.tobytes())

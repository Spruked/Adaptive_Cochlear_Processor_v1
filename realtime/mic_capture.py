# Copyright (c) 2026 Adaptive Cochlear Processor Team
# Licensed under the MIT License. See LICENSE file for details.

"""
MicCapture: Push-to-buffer live audio pipeline for ACP.
Records 3-second chunks → temp wav → ACP.hear() → repeat.
No threading. No async. No streaming decode. Just chunked hearing.
"""
import sounddevice as sd
import numpy as np
import soundfile as sf
from pathlib import Path
import time
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from acp_1_0.acp.core import AdaptiveCochlearProcessor

class MicCapture:
    """
    Simple live audio capture for ACP.
    Records fixed-length chunks, processes sequentially.
    """

    def __init__(self, sample_rate=16000, chunk_duration=3.0):
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.chunk_samples = int(sample_rate * chunk_duration)

        # ACP processor (reuse same instance for continuous learning)
        self.processor = AdaptiveCochlearProcessor(
            skg_path="skg/hearing_live.json"  # Separate SKG for live data
        )

        # Buffer file (overwritten each chunk)
        self.buffer_path = Path("audio_cache/buffer.wav")
        self.buffer_path.parent.mkdir(exist_ok=True)

        print(f"🎤 MicCapture initialized")
        print(f"   Sample rate: {sample_rate} Hz")
        print(f"   Chunk duration: {chunk_duration} sec")
        print(f"   Buffer path: {self.buffer_path}")
        print(f"\nPress Ctrl+C to stop...\n")

    def record_chunk(self) -> np.ndarray:
        """Record one chunk of audio from default microphone"""
        print(f"🔴 Recording {self.chunk_duration} seconds...")

        # Record audio
        audio = sd.rec(
            self.chunk_samples,
            samplerate=self.sample_rate,
            channels=1,
            dtype=np.float32,
            blocking=True  # Wait until complete
        )

        print(f"✅ Recording complete ({len(audio)} samples)")
        return audio

    def process_chunk(self, audio: np.ndarray):
        """Save chunk to file, call ACP, print results"""
        # Save to temporary file
        audio = audio.squeeze()  # Ensure 1D array
        sf.write(self.buffer_path, audio, self.sample_rate)

        # Process with ACP (includes perception, decode, cognition, learning)
        result = self.processor.hear(
            str(self.buffer_path),
            context={
                "source": "live_mic",
                "timestamp": time.time()
            }
        )

        # Print summary
        print(f"\n📝 Result:")
        print(f"   Transcript: \"{result['transcript']}\"")
        print(f"   Confidence: {result['confidence']:.2f}")
        print(f"   Corrections: {len(result.get('corrections', []))}")
        print(f"   Phoneme mastery: {result['learning_snapshot']['avg_phoneme_mastery']:.2f}")
        print(f"-" * 50)

        return result

    def run(self, max_chunks=None):
        """Main loop: record → process → repeat"""
        chunk_count = 0
        try:
            while True:
                if max_chunks and chunk_count >= max_chunks:
                    print(f"\n🛑 Reached max chunks ({max_chunks})")
                    break

                # Record
                audio = self.record_chunk()

                # Process
                self.process_chunk(audio)

                # Brief pause between chunks (optional)
                time.sleep(0.1)
                chunk_count += 1

        except KeyboardInterrupt:
            print("\n\n🛑 Stopped by user")
            self._print_summary()
        except Exception as e:
            print(f"\n❌ Error: {e}")
            self._print_summary()

    def _print_summary(self):
        """Print final processing statistics"""
        print(f"\n📊 Session Summary:")
        print(f"   Total transcriptions: {self.processor.processing_count}")
        print(f"   Final phoneme mastery: {self.processor.learning.get_state()['avg_phoneme_mastery']:.2f}")
        print(f"   SKG saved to: {self.processor.learning.path}")

def main():
    """Entry point: create and run MicCapture"""
    import sys

    # Parse command line args
    max_chunks = None
    if len(sys.argv) > 1:
        try:
            max_chunks = int(sys.argv[1])
        except ValueError:
            print(f"❌ Invalid max_chunks: {sys.argv[1]}")
            return

    capture = MicCapture(sample_rate=16000, chunk_duration=3.0)
    capture.run(max_chunks=max_chunks)

if __name__ == "__main__":
    main()

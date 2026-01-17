# Copyright (c) 2026 Adaptive Cochlear Processor Team
# Licensed under the MIT License. See LICENSE file for details.

"""
KeyboardGate: Push-to-talk wrapper for MicCapture.
Hold SPACE to record → release to process → see results.
No threading. No async. Just blocking keyboard input.
"""
import sys
from pathlib import Path
import time

# Cross-platform keyboard library
try:
    import keyboard
except ImportError:
    print("❌ Install keyboard: pip install keyboard")
    sys.exit(1)

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from realtime.mic_capture import MicCapture

class KeyboardGate:
    """
    Push-to-talk controller for ACP.
    SPACE = record, 'q' = quit, Ctrl+C = force quit
    """

    def __init__(self, chunk_duration=3.0):
        self.capture = MicCapture(chunk_duration=chunk_duration)
        self.is_recording = False

        print("\n🎤 Push-to-Talk Mode")
        print("   Hold SPACE to record")
        print("   Release to process")
        print("   Press 'q' to quit")
        print("   Ctrl+C to force stop\n")

        # Set up hotkeys
        keyboard.on_press_key("space", self._start_recording)
        keyboard.on_release_key("space", self._stop_recording)
        keyboard.on_press_key("q", self._quit)

    def _start_recording(self, event):
        """SPACE pressed - start recording"""
        if not self.is_recording:
            self.is_recording = True
            print("\n🔴 RECORDING... (release SPACE to process)", end=" ")

    def _stop_recording(self, event):
        """SPACE released - stop and process"""
        if self.is_recording:
            self.is_recording = False
            print("✓ STOPPED")

            # Process the chunk
            audio = self.capture.record_chunk()
            self.capture.process_chunk(audio)

    def _quit(self, event):
        """'q' pressed - graceful exit"""
        print("\n\n🛑 Quitting...")
        self._cleanup()
        sys.exit(0)

    def _cleanup(self):
        """Remove keyboard hooks before exit"""
        try:
            keyboard.unhook_all()
        except:
            pass

    def run(self):
        """Main loop: wait for keyboard events"""
        try:
            print("⌨️  Ready. Hold SPACE to begin...")
            keyboard.wait()  # Blocks until unhooked
        except KeyboardInterrupt:
            print("\n\n🛑 Force quit (Ctrl+C)")
            self._cleanup()
            sys.exit(0)

def main():
    """Entry point"""
    gate = KeyboardGate(chunk_duration=3.0)
    gate.run()

if __name__ == "__main__":
    main()

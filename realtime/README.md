# Realtime Audio Capture for ACP

Live microphone input → ACP processing → continuous learning.

## Setup

```bash
# Install audio dependencies
pip install sounddevice

# On Windows, you might also need:
pip install PyAudio  # Alternative audio backend
```

## Usage

```bash
# Start live capture (3-second chunks)
python realtime/mic_capture.py

# Speak into your microphone
# Each 3-second chunk will be processed by ACP
# Results show transcript, confidence, corrections, and learning progress
# Press Ctrl+C to stop
```

## What It Does

1. **Records** 3-second audio chunks from your microphone
2. **Saves** each chunk to `audio_cache/buffer.wav`
3. **Processes** through ACP (perception → decode → cognition → learning)
4. **Displays** results in real-time
5. **Learns** continuously from corrections
6. **Saves** learning progress to `skg/hearing_live.json`

## Output Example

```
🎤 MicCapture initialized
   Sample rate: 16000 Hz
   Chunk duration: 3.0 sec
   Buffer path: audio_cache\buffer.wav

Press Ctrl+C to stop...

🔴 Recording 3.0 seconds...
✅ Recording complete (48000 samples)

📝 Result:
   Transcript: "hello world this is a test"
   Confidence: 0.85
   Corrections: 1
   Phoneme mastery: 0.67
--------------------------------------------------
```

## Monitoring Learning

While capture is running, monitor progress in another terminal:

```bash
# Watch metrics in real-time
watch -n 5 'python -c "
from acp_1_0.acp.metrics import MetricsTracker
mt = MetricsTracker(\"acp_metrics_live.jsonl\")
summary = mt.get_summary(last_n=10)
print(f\"Confidence: {summary[\"confidence_trend\"][\"end\"]:.2f} (Δ {summary[\"confidence_trend\"][\"delta\"]:+.2f})\")
print(f\"Corrections/100w: {summary[\"corrections_trend\"][\"end\"]:.1f} (Δ {summary[\"corrections_trend\"][\"delta\"]:+.1f})\")
print(f\"Learning active: {summary[\"learning_active\"]}\")
"'
```

## Files Created

- `skg/hearing_live.json` - Learning knowledge graph
- `acp_metrics_live.jsonl` - Detailed metrics log
- `audio_cache/buffer.wav` - Temporary audio buffer

## Troubleshooting

**No microphone detected:**
- Check microphone permissions
- Try different audio device: `python -c "import sounddevice as sd; print(sd.query_devices())"`

**Import errors:**
- Ensure `sounddevice` is installed
- On Windows, try: `pip install sounddevice[portaudio]`

**Performance issues:**
- Reduce `chunk_duration` for faster processing
- Lower `sample_rate` if needed (but 16000 is optimal for ACP)
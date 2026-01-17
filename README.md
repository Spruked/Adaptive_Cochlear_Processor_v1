# ACP 1.0 - Adaptive Cochlear Processor

**Perception-first STT that learns from its mistakes.**

Unlike Whisper, ACP models human hearing: it mishears, corrects itself,
and gets better over time.

## Quick Start
```bash
pip install acp-1.0
```

```python
from acp import AdaptiveCochlearProcessor

processor = AdaptiveCochlearProcessor()
result = processor.hear("podcast.wav", context={"topic": "AI"})

print(result["final_transcript"])
# "The future of AI is machine learning"
print(f"Made {len(result['corrections'])} corrections")
# "Made 3 corrections"
```

## Architecture
- **Perception**: Human ear simulation (frequency, attention, fatigue)
- **Cognition**: Error detection & context-based correction
- **Learning**: Persistent SKG memory (phoneme & speaker mastery)

## For Developers
```bash
# Install with teaching tools
pip install acp-1.0[dev]

# Generate training data from Whisper
python -m acp.backend.whisper_teacher analyze --input audio/
# → Saves spectrograms, word alignments, failure patterns
```

## Performance
- **CPU-optimized**: ~5sec/minute on Core i5
- **Memory**: <50MB (no giant models)
- **Learning**: SKG updates in real-time

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
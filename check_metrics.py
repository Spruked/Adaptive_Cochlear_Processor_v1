# Copyright (c) 2026 Adaptive Cochlear Processor Team
# Licensed under the MIT License. See LICENSE file for details.

#!/usr/bin/env python3
"""
Check current learning metrics from live capture
"""
import sys
from pathlib import Path

# Add paths for imports
sys.path.append(str(Path(__file__).parent.parent))

try:
    from acp_1_0.acp.metrics import MetricsTracker

    # Check if SKG has simulated samples flag
    skg_path = Path("skg/hearing_live.json")
    if skg_path.exists():
        import json
        skg = json.loads(skg_path.read_text())
        simulated_count = skg.get("metadata", {}).get("simulated_samples", 0)
        
        if simulated_count > 0:
            print(f"⚠️  WARNING: {simulated_count} simulated samples detected!")
            print("   Real learning cannot occur until Whisper is working.")
            print("   Fix: pip install numpy==1.24.3 --no-binary numpy\n")

    # Check live metrics
    mt = MetricsTracker("acp_metrics_live.jsonl")
    summary = mt.get_summary(last_n=10)

    if "error" in summary:
        print("📊 No metrics yet - run mic capture first")
        print("python realtime/mic_capture.py")
    else:
        print("📊 Current Learning Status:")
        print(f"   Total transcriptions: {summary['total_transcriptions']}")
        print(f"   Last {summary['last_n']} chunks:")
        print(f"   Confidence: {summary['confidence_trend']['end']:.2f} (Δ {summary['confidence_trend']['delta']:+.2f})")
        print(f"   Corrections/100w: {summary['corrections_trend']['end']:.1f} (Δ {summary['corrections_trend']['delta']:+.1f})")
        print(f"   Learning active: {summary['learning_active']}")
        print(f"   Phoneme mastery: {summary['latest_mastery']['phoneme']:.3f}")
        print(f"   Speaker mastery: {summary['latest_mastery']['speaker']:.3f}")

        # Learning indicators
        if summary['learning_active']:
            print("   ✅ System is learning!")
        else:
            print("   ⚠️  No learning detected yet - check for simulated data or low correction rate")

except Exception as e:
    print(f"❌ Error checking metrics: {e}")

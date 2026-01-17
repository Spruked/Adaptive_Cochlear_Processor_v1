# Copyright (c) 2026 Adaptive Cochlear Processor Team
# Licensed under the MIT License. See LICENSE file for details.

"""
ACP Voice Module: Placeholder for TTS integration.
This will be implemented when ACP TTS is ready.
"""

class ACPSynthesisEngine:
    """
    Placeholder for ACP-based voice synthesis.
    Currently raises NotImplementedError.
    """
    
    def __init__(self, skg_path="skg/hearing.json"):
        self.skg_path = skg_path
        # TODO: Implement ACP voice synthesis
    
    def speak(self, text: str, speaker_id: str = None, **kwargs):
        """
        Placeholder method.
        """
        raise NotImplementedError("ACP Voice synthesis not yet implemented")

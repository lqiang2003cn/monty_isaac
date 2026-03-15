"""Vision foundation model wrappers for the turntable scanning pipeline.

Provides:
- SAM2Segmenter: object segmentation via Meta's Segment Anything 2
- VGGTProvider: camera poses + depth maps from RGB via Meta's VGGT

The models run in a separate container (vision_comp) and are accessed
via a TCP JSON-lines bridge (``_bridge.VisionBridge``).
"""

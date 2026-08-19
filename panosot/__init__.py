"""PanoSOT baseline package."""

from .factory import build_tracker
from .tracker import PanoSOTTracker, TrackerConfig

__all__ = ["PanoSOTTracker", "TrackerConfig", "build_tracker"]
"""PanoSOT public API."""

from .factory import build_tracker
from .siamx360 import ERPRegionCropper, SiamX360Config, SiamX360Tracker
from .tracker import PanoSOTTracker, TrackerConfig

__all__ = [
    "ERPRegionCropper",
    "PanoSOTTracker",
    "SiamX360Config",
    "SiamX360Tracker",
    "TrackerConfig",
    "build_tracker",
]

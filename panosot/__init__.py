"""PanoSOT baseline package."""

from .factory import build_tracker
from .tracker import PanoSOTTracker, TrackerConfig

try:
    from .ostrack_tracker import PanoOSTrackTracker, build_ostrack_tracker
except ImportError:  # torch is optional for the handcrafted pipeline
    PanoOSTrackTracker = None  # type: ignore[assignment]
    build_ostrack_tracker = None  # type: ignore[assignment]

__all__ = [
    "PanoSOTTracker",
    "TrackerConfig",
    "build_tracker",
    "PanoOSTrackTracker",
    "build_ostrack_tracker",
]

"""Configuration for diarization models."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class DiarizationConfig:
    """Configuration for speaker diarization."""

    # Model selection
    model_name: str = "pyannote/speaker-diarization-3.1"
    use_auth_token: Optional[bool] = None

    # Processing parameters
    min_speakers: Optional[int] = None
    max_speakers: Optional[int] = None

    # Segmentation parameters
    segmentation_duration: float = 15.0  # seconds
    overlap: float = 0.5  # overlap ratio between segments

    # Clustering parameters
    clustering_threshold: Optional[float] = None

    # Device settings
    device: str = "auto"  # "auto", "cpu", "cuda"

    # Merging parameters
    segment_merge_threshold: float = (
        2.5  # seconds gap for merging same-speaker segments
    )

    # Output settings
    return_embeddings: bool = False

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.segmentation_duration <= 0:
            raise ValueError("segmentation_duration must be positive")

        if not 0 <= self.overlap < 1:
            raise ValueError("overlap must be in range [0, 1)")

        if self.min_speakers is not None and self.min_speakers <= 0:
            raise ValueError("min_speakers must be positive")

        if self.max_speakers is not None and self.max_speakers <= 0:
            raise ValueError("max_speakers must be positive")

        if (
            self.min_speakers is not None
            and self.max_speakers is not None
            and self.min_speakers > self.max_speakers
        ):
            raise ValueError("min_speakers cannot be greater than max_speakers")

"""Type definitions for speaker diarization."""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from enum import Enum


class DiarizationStatus(Enum):
    """Status of diarization processing."""

    PENDING = "pending"
    LOADING = "loading"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class SpeakerSegment:
    """Represents a segment of speech from a single speaker."""

    start: float
    end: float
    speaker: str
    confidence: Optional[float] = None

    def __post_init__(self):
        """Validate segment data."""
        if self.start < 0:
            raise ValueError("Start time cannot be negative")
        if self.end <= self.start:
            raise ValueError("End time must be greater than start time")
        if not self.speaker:
            raise ValueError("Speaker identifier cannot be empty")

    @property
    def duration(self) -> float:
        """Get segment duration."""
        return self.end - self.start

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "start": self.start,
            "end": self.end,
            "duration": self.duration,
            "speaker": self.speaker,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SpeakerSegment":
        """Create from dictionary."""
        return cls(
            start=data["start"],
            end=data["end"],
            speaker=data["speaker"],
            confidence=data.get("confidence"),
        )


@dataclass
class DiarizationResult:
    """Complete diarization result for an audio file."""

    segments: List[SpeakerSegment]
    speakers: List[str]
    num_speakers: int
    total_duration: float
    status: DiarizationStatus = DiarizationStatus.COMPLETED
    error: Optional[str] = None

    def __post_init__(self):
        """Validate result data."""
        if not self.segments:
            raise ValueError("Segments list cannot be empty")
        if self.num_speakers <= 0:
            raise ValueError("Number of speakers must be positive")
        if self.total_duration <= 0:
            raise ValueError("Total duration must be positive")
        if len(self.speakers) != self.num_speakers:
            raise ValueError("Speakers list length must match num_speakers")

    def get_segments_by_speaker(self, speaker: str) -> List[SpeakerSegment]:
        """Get all segments for a specific speaker."""
        return [seg for seg in self.segments if seg.speaker == speaker]

    def get_speaker_total_time(self, speaker: str) -> float:
        """Get total speaking time for a speaker."""
        return sum(seg.duration for seg in self.get_segments_by_speaker(speaker))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "segments": [seg.to_dict() for seg in self.segments],
            "speakers": self.speakers,
            "num_speakers": self.num_speakers,
            "total_duration": self.total_duration,
            "status": self.status.value,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DiarizationResult":
        """Create from dictionary."""
        return cls(
            segments=[SpeakerSegment.from_dict(seg) for seg in data["segments"]],
            speakers=data["speakers"],
            num_speakers=data["num_speakers"],
            total_duration=data["total_duration"],
            status=DiarizationStatus(data.get("status", "completed")),
            error=data.get("error"),
        )


@dataclass
class DiarizationProgress:
    """Progress information during diarization."""

    stage: str
    progress: float  # 0.0 to 1.0
    message: Optional[str] = None
    current_step: Optional[int] = None
    total_steps: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "stage": self.stage,
            "progress": self.progress,
            "message": self.message,
            "current_step": self.current_step,
            "total_steps": self.total_steps,
        }


# Type alias for progress callback
from typing import Callable

ProgressCallback = Optional[Callable[[DiarizationProgress], None]]

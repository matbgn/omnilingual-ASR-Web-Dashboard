"""Smart chunking for building speaker turns from diarization segments."""

import logging
import os
import tempfile
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass

from .diarization_types import SpeakerSegment, DiarizationResult
from .config import DiarizationConfig

logger = logging.getLogger(__name__)

try:
    import torchaudio
    import torch

    TORCHAUDIO_AVAILABLE = True
except ImportError:
    TORCHAUDIO_AVAILABLE = False
    torchaudio = None
    torch = None


@dataclass
class SpeakerTurn:
    """Represents a continuous speech segment from a single speaker."""

    start: float
    end: float
    speaker: str
    original_segments: List[SpeakerSegment]  # The segments that were merged

    def __post_init__(self):
        """Validate turn data."""
        if self.start < 0:
            raise ValueError("Start time cannot be negative")
        if self.end <= self.start:
            raise ValueError("End time must be greater than start time")
        if not self.speaker:
            raise ValueError("Speaker identifier cannot be empty")
        if not self.original_segments:
            raise ValueError("Original segments list cannot be empty")

    @property
    def duration(self) -> float:
        """Get turn duration."""
        return self.end - self.start

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "start": self.start,
            "end": self.end,
            "duration": self.duration,
            "speaker": self.speaker,
            "original_segments": [seg.to_dict() for seg in self.original_segments],
            "segment_count": len(self.original_segments),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SpeakerTurn":
        """Create from dictionary."""
        original_segments = [
            SpeakerSegment.from_dict(seg) for seg in data["original_segments"]
        ]
        return cls(
            start=data["start"],
            end=data["end"],
            speaker=data["speaker"],
            original_segments=original_segments,
        )


class SmartChunkBuilder:
    """Builds speaker turns from diarization segments and extracts audio chunks."""

    def __init__(self, config: Optional[DiarizationConfig] = None):
        """Initialize the SmartChunkBuilder.

        Args:
            config: Diarization configuration. If None, uses defaults.
        """
        self.config = config or DiarizationConfig()
        self._validate_config()

    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        if self.config.segment_merge_threshold < 0:
            raise ValueError("segment_merge_threshold cannot be negative")
        if self.config.segmentation_duration <= 0:
            raise ValueError("segmentation_duration must be positive")

    def build_speaker_turns(
        self,
        diarization_result: DiarizationResult,
        min_turn_duration: float = 1.0,
        gap_threshold: Optional[float] = None,
    ) -> List[SpeakerTurn]:
        """Merge diarization segments into speaker turns.

        Args:
            diarization_result: Result from diarization service
            min_turn_duration: Minimum duration for a turn (seconds)
            gap_threshold: Maximum gap between same-speaker segments to merge (seconds).
                          If None, uses config.segment_merge_threshold.

        Returns:
            List of SpeakerTurn objects
        """
        if not diarization_result.segments:
            logger.warning("No segments in diarization result")
            return []

        # Use config value if gap_threshold not provided
        if gap_threshold is None:
            gap_threshold = getattr(self.config, "segment_merge_threshold", 0.5)

        # Sort segments by start time
        sorted_segments = sorted(diarization_result.segments, key=lambda s: s.start)

        turns = []
        current_segments = [sorted_segments[0]]

        for i in range(1, len(sorted_segments)):
            current_segment = sorted_segments[i]
            last_segment = current_segments[-1]

            # Check if we should merge
            if (
                current_segment.speaker == last_segment.speaker
                and (current_segment.start - last_segment.end) <= gap_threshold
            ):
                # Same speaker within gap threshold - merge
                current_segments.append(current_segment)
            else:
                # Different speaker or gap too large - finalize current turn
                turn = self._create_turn_from_segments(
                    current_segments, min_turn_duration
                )
                if turn:
                    turns.append(turn)

                # Start new turn
                current_segments = [current_segment]

        # Don't forget the last turn
        if current_segments:
            turn = self._create_turn_from_segments(current_segments, min_turn_duration)
            if turn:
                turns.append(turn)

        logger.info(
            f"Built {len(turns)} speaker turns from {len(sorted_segments)} segments"
        )
        return turns

    def _create_turn_from_segments(
        self, segments: List[SpeakerSegment], min_duration: float
    ) -> Optional[SpeakerTurn]:
        """Create a speaker turn from a list of segments.

        Args:
            segments: List of segments to merge
            min_duration: Minimum duration for the turn

        Returns:
            SpeakerTurn if duration meets minimum, None otherwise
        """
        if not segments:
            return None

        start = min(seg.start for seg in segments)
        end = max(seg.end for seg in segments)
        speaker = segments[0].speaker  # All segments should have same speaker
        duration = end - start

        # Check minimum duration
        if duration < min_duration:
            logger.debug(
                f"Skipping turn for {speaker}: duration {duration:.2f}s < {min_duration}s"
            )
            return None

        return SpeakerTurn(
            start=start,
            end=end,
            speaker=speaker,
            original_segments=segments.copy(),
        )

    def extract_turn_audio(
        self,
        audio_path: str,
        turn: SpeakerTurn,
        output_path: Optional[str] = None,
        sample_rate: Optional[int] = None,
        padding: float = 0.2,
    ) -> str:
        """Extract audio for a specific speaker turn.

        Args:
            audio_path: Path to the original audio file
            turn: Speaker turn to extract
            output_path: Output path for extracted audio. If None, creates temp file.
            sample_rate: Target sample rate. If None, uses original.
            padding: Padding in seconds to add before and after the turn (default: 0.2s)

        Returns:
            Path to the extracted audio file

        Raises:
            ImportError: If torchaudio is not available
            ValueError: If audio file is invalid
            RuntimeError: If audio extraction fails
        """
        if not TORCHAUDIO_AVAILABLE:
            raise ImportError(
                "torchaudio is required for audio extraction. "
                "Install with: pip install torchaudio"
            )

        if not os.path.exists(audio_path):
            raise ValueError(f"Audio file not found: {audio_path}")

        try:
            # Load audio
            waveform, original_sample_rate = torchaudio.load(audio_path)

            # Convert to mono if needed
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            # Convert sample rate if needed
            if sample_rate and sample_rate != original_sample_rate:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=original_sample_rate, new_freq=sample_rate
                )
                waveform = resampler(waveform)
                adjusted_sample_rate = sample_rate
            else:
                adjusted_sample_rate = original_sample_rate

            # Calculate sample indices for the turn with padding
            start_time = max(0, turn.start - padding)
            end_time = turn.end + padding
            
            start_sample = int(start_time * adjusted_sample_rate)
            end_sample = int(end_time * adjusted_sample_rate)

            # Validate indices
            if start_sample < 0:
                start_sample = 0
            if end_sample > waveform.shape[1]:
                end_sample = waveform.shape[1]
            if start_sample >= end_sample:
                raise ValueError(f"Invalid turn time range: {turn.start}-{turn.end}")

            # Extract the audio segment
            turn_waveform = waveform[:, start_sample:end_sample]

            # Generate output path if not provided
            if output_path is None:
                temp_dir = tempfile.gettempdir()
                file_basename = os.path.splitext(os.path.basename(audio_path))[0]
                output_path = os.path.join(
                    temp_dir,
                    f"{file_basename}_turn_{turn.speaker}_{turn.start:.1f}-{turn.end:.1f}.wav",
                )

            # Save the extracted audio
            torchaudio.save(output_path, turn_waveform, adjusted_sample_rate)

            logger.debug(f"Extracted turn audio: {turn.duration:.2f}s -> {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Failed to extract turn audio: {e}")
            raise RuntimeError(f"Audio extraction failed: {e}") from e

    def analyze_turns(self, turns: List[SpeakerTurn]) -> Dict[str, Any]:
        """Analyze speaker turns and provide statistics.

        Args:
            turns: List of speaker turns

        Returns:
            Dictionary with turn analysis
        """
        if not turns:
            return {
                "total_turns": 0,
                "total_duration": 0.0,
                "speakers": [],
                "speaker_stats": {},
            }

        # Group turns by speaker
        speaker_turns = {}
        for turn in turns:
            if turn.speaker not in speaker_turns:
                speaker_turns[turn.speaker] = []
            speaker_turns[turn.speaker].append(turn)

        # Calculate statistics
        speaker_stats = {}
        for speaker, speaker_turn_list in speaker_turns.items():
            durations = [turn.duration for turn in speaker_turn_list]
            speaker_stats[speaker] = {
                "turn_count": len(speaker_turn_list),
                "total_time": sum(durations),
                "avg_turn_duration": sum(durations) / len(durations),
                "min_turn_duration": min(durations),
                "max_turn_duration": max(durations),
            }

        total_duration = sum(turn.duration for turn in turns)

        return {
            "total_turns": len(turns),
            "total_duration": total_duration,
            "speakers": list(speaker_turns.keys()),
            "num_speakers": len(speaker_turns.keys()),
            "speaker_stats": speaker_stats,
            "avg_turn_duration": total_duration / len(turns),
        }

    def validate_turns(self, turns: List[SpeakerTurn]) -> List[str]:
        """Validate speaker turns and return list of issues.

        Args:
            turns: List of speaker turns to validate

        Returns:
            List of validation issues (empty if valid)
        """
        issues = []

        if not turns:
            return ["No turns provided"]

        # Check for overlapping turns from same speaker
        for i, turn1 in enumerate(turns):
            for j, turn2 in enumerate(turns[i + 1 :], i + 1):
                if turn1.speaker == turn2.speaker and not (
                    turn1.end <= turn2.start or turn2.end <= turn1.start
                ):
                    issues.append(
                        f"Overlapping turns for {turn1.speaker}: "
                        f"{turn1.start:.1f}-{turn1.end:.1f} and {turn2.start:.1f}-{turn2.end:.1f}"
                    )

        # Check for very short turns
        min_duration = 0.5  # Minimum reasonable turn duration
        for turn in turns:
            if turn.duration < min_duration:
                issues.append(
                    f"Very short turn for {turn.speaker}: {turn.duration:.2f}s "
                    f"({turn.start:.1f}-{turn.end:.1f})"
                )

        # Check for gaps that might indicate missing segments
        if len(turns) > 1:
            sorted_turns = sorted(turns, key=lambda t: t.start)
            large_gaps = []
            for i in range(len(sorted_turns) - 1):
                current = sorted_turns[i]
                next_turn = sorted_turns[i + 1]
                gap = next_turn.start - current.end
                if gap > 5.0:  # Gap larger than 5 seconds
                    large_gaps.append((current, next_turn, gap))

            if large_gaps:
                issues.append(f"Found {len(large_gaps)} large gaps between turns (>5s)")

        return issues

    def get_supported_formats(self) -> List[str]:
        """Get list of supported audio formats for extraction."""
        if not TORCHAUDIO_AVAILABLE:
            return []

        # Common formats supported by torchaudio
        return [".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".wma"]

    def __call__(
        self,
        diarization_result: DiarizationResult,
        min_turn_duration: float = 1.0,
        gap_threshold: Optional[float] = None,
    ) -> List[SpeakerTurn]:
        """Convenience method to make the builder callable.

        Args:
            diarization_result: Result from diarization service
            min_turn_duration: Minimum duration for a turn (seconds)
            gap_threshold: Maximum gap between same-speaker segments to merge

        Returns:
            List of SpeakerTurn objects
        """
        return self.build_speaker_turns(
            diarization_result, min_turn_duration, gap_threshold
        )

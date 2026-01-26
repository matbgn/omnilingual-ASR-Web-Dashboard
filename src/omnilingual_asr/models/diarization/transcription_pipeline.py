"""Transcription pipeline for processing speaker turns efficiently."""

import gc
import logging
import os
import tempfile
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable, Generator

from .chunking import SmartChunkBuilder, SpeakerTurn
from .diarization_types import DiarizationResult

logger = logging.getLogger(__name__)


@dataclass
class TranscriptionSegment:
    """Result of transcribing a single speaker turn."""

    speaker: str
    start: float
    end: float
    duration: float
    text: str
    turn_id: int
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        result = {
            "speaker": self.speaker,
            "start": self.start,
            "end": self.end,
            "duration": self.duration,
            "text": self.text,
            "turn_id": self.turn_id,
        }
        if self.error:
            result["error"] = self.error
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TranscriptionSegment":
        """Create from dictionary."""
        return cls(
            speaker=data["speaker"],
            start=data["start"],
            end=data["end"],
            duration=data["duration"],
            text=data["text"],
            turn_id=data["turn_id"],
            error=data.get("error"),
        )


@dataclass
class TranscriptionProgress:
    """Progress update during transcription."""

    stage: str  # "extracting", "transcribing", "complete"
    current_turn: int
    total_turns: int
    speaker: Optional[str] = None
    message: Optional[str] = None

    @property
    def progress_value(self) -> float:
        """Calculate progress as a float from 0.0 to 1.0."""
        if self.total_turns == 0:
            return 0.0
        return self.current_turn / self.total_turns

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "stage": self.stage,
            "progress": self.progress_value,
            "current_turn": self.current_turn,
            "total_turns": self.total_turns,
            "speaker": self.speaker,
            "message": self.message,
        }


@dataclass
class TranscriptionResult:
    """Complete transcription result for all speaker turns."""

    segments: List[TranscriptionSegment]
    full_transcript: str
    num_speakers: int
    total_duration: float
    speakers: List[str]
    successful_turns: int = 0
    failed_turns: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "segments": [seg.to_dict() for seg in self.segments],
            "full_transcript": self.full_transcript,
            "num_speakers": self.num_speakers,
            "total_duration": self.total_duration,
            "speakers": self.speakers,
            "successful_turns": self.successful_turns,
            "failed_turns": self.failed_turns,
        }


ProgressCallback = Callable[[TranscriptionProgress], None]
SegmentCallback = Callable[[TranscriptionSegment], None]


class SpeakerTurnTranscriptionPipeline:
    """Orchestrates efficient transcription of speaker turns.

    This pipeline handles:
    - Audio extraction for each turn
    - Memory management (cleanup between operations)
    - Per-turn error handling (continues on failure)
    - Progress reporting via callbacks
    - Optional batching for efficiency
    """

    def __init__(
        self,
        asr_pipeline,
        chunk_builder: Optional[SmartChunkBuilder] = None,
        batch_size: int = 1,
        cleanup_temp_files: bool = True,
    ):
        """Initialize the transcription pipeline.

        Args:
            asr_pipeline: ASR inference pipeline with transcribe() method
            chunk_builder: SmartChunkBuilder for audio extraction. Uses default if None.
            batch_size: Number of turns to batch for transcription (default 1 for memory)
            cleanup_temp_files: Whether to delete temp audio files after use
        """
        self.asr_pipeline = asr_pipeline
        self.chunk_builder = chunk_builder or SmartChunkBuilder()
        self.batch_size = batch_size
        self.cleanup_temp_files = cleanup_temp_files
        self._temp_files: List[str] = []

    def transcribe_turns(
        self,
        audio_path: str,
        turns: List[SpeakerTurn],
        lang_code: Optional[str] = None,
        progress_callback: Optional[ProgressCallback] = None,
        segment_callback: Optional[SegmentCallback] = None,
    ) -> List[TranscriptionSegment]:
        """Transcribe all speaker turns from an audio file.

        Args:
            audio_path: Path to the original audio file
            turns: List of speaker turns to transcribe
            lang_code: Language code for transcription
            progress_callback: Called with progress updates
            segment_callback: Called when each segment completes

        Returns:
            List of TranscriptionSegment objects
        """
        segments = []
        total_turns = len(turns)

        logger.info(f"Starting transcription of {total_turns} speaker turns")

        try:
            for i, turn in enumerate(turns):
                current_turn = i + 1

                # Report progress - extracting
                if progress_callback:
                    progress_callback(
                        TranscriptionProgress(
                            stage="extracting",
                            current_turn=current_turn,
                            total_turns=total_turns,
                            speaker=turn.speaker,
                            message=f"Extracting audio for turn {current_turn}/{total_turns}",
                        )
                    )

                # Extract audio for this turn
                try:
                    turn_audio_path = self.chunk_builder.extract_turn_audio(
                        audio_path, turn
                    )
                    self._temp_files.append(turn_audio_path)
                except Exception as e:
                    logger.error(f"Failed to extract audio for turn {i}: {e}")
                    segment = TranscriptionSegment(
                        speaker=turn.speaker,
                        start=turn.start,
                        end=turn.end,
                        duration=turn.duration,
                        text="[Audio extraction failed]",
                        turn_id=i,
                        error=str(e),
                    )
                    segments.append(segment)
                    if segment_callback:
                        segment_callback(segment)
                    continue

                # Report progress - transcribing
                if progress_callback:
                    progress_callback(
                        TranscriptionProgress(
                            stage="transcribing",
                            current_turn=current_turn,
                            total_turns=total_turns,
                            speaker=turn.speaker,
                            message=f"Transcribing turn {current_turn}/{total_turns}: {turn.speaker} ({turn.duration:.1f}s)",
                        )
                    )

                # Transcribe the turn
                segment = self._transcribe_single_turn(
                    turn_audio_path, turn, i, lang_code
                )
                segments.append(segment)

                if segment_callback:
                    segment_callback(segment)

                # Cleanup temp file immediately if configured
                if self.cleanup_temp_files:
                    self._cleanup_file(turn_audio_path)

            # Report completion
            if progress_callback:
                progress_callback(
                    TranscriptionProgress(
                        stage="complete",
                        current_turn=total_turns,
                        total_turns=total_turns,
                        message=f"Transcription complete: {total_turns} turns processed",
                    )
                )

        finally:
            # Ensure all temp files are cleaned up
            self._cleanup_all_temp_files()

        logger.info(f"Transcription complete: {len(segments)} segments")
        return segments

    def transcribe_turns_streaming(
        self,
        audio_path: str,
        turns: List[SpeakerTurn],
        lang_code: Optional[str] = None,
    ) -> Generator[TranscriptionSegment | TranscriptionProgress, None, None]:
        """Transcribe speaker turns and yield results as a stream.

        This is useful for SSE streaming where you want to emit events
        as they happen rather than through callbacks.

        Args:
            audio_path: Path to the original audio file
            turns: List of speaker turns to transcribe
            lang_code: Language code for transcription

        Yields:
            TranscriptionProgress or TranscriptionSegment objects
        """
        total_turns = len(turns)

        logger.info(f"Starting streaming transcription of {total_turns} speaker turns")

        try:
            for i, turn in enumerate(turns):
                current_turn = i + 1

                # Yield progress - extracting
                yield TranscriptionProgress(
                    stage="extracting",
                    current_turn=current_turn,
                    total_turns=total_turns,
                    speaker=turn.speaker,
                    message=f"Extracting audio for turn {current_turn}/{total_turns}",
                )

                # Extract audio for this turn
                try:
                    turn_audio_path = self.chunk_builder.extract_turn_audio(
                        audio_path, turn
                    )
                    self._temp_files.append(turn_audio_path)
                except Exception as e:
                    logger.error(f"Failed to extract audio for turn {i}: {e}")
                    segment = TranscriptionSegment(
                        speaker=turn.speaker,
                        start=turn.start,
                        end=turn.end,
                        duration=turn.duration,
                        text="[Audio extraction failed]",
                        turn_id=i,
                        error=str(e),
                    )
                    yield segment
                    continue

                # Yield progress - transcribing
                yield TranscriptionProgress(
                    stage="transcribing",
                    current_turn=current_turn,
                    total_turns=total_turns,
                    speaker=turn.speaker,
                    message=f"Transcribing turn {current_turn}/{total_turns}: {turn.speaker} ({turn.duration:.1f}s)",
                )

                # Transcribe the turn
                segment = self._transcribe_single_turn(
                    turn_audio_path, turn, i, lang_code
                )
                yield segment

                # Cleanup temp file immediately
                if self.cleanup_temp_files:
                    self._cleanup_file(turn_audio_path)

            # Yield completion progress
            yield TranscriptionProgress(
                stage="complete",
                current_turn=total_turns,
                total_turns=total_turns,
                message=f"Transcription complete: {total_turns} turns processed",
            )

        finally:
            # Ensure all temp files are cleaned up
            self._cleanup_all_temp_files()

    def _transcribe_single_turn(
        self,
        audio_path: str,
        turn: SpeakerTurn,
        turn_id: int,
        lang_code: Optional[str] = None,
    ) -> TranscriptionSegment:
        """Transcribe a single speaker turn.

        Args:
            audio_path: Path to the extracted turn audio
            turn: Speaker turn being transcribed
            turn_id: Index of this turn
            lang_code: Language code for transcription

        Returns:
            TranscriptionSegment with result
        """
        try:
            transcriptions = self.asr_pipeline.transcribe(
                [audio_path], lang=[lang_code], batch_size=1
            )

            if transcriptions:
                turn_text = transcriptions[0]
            else:
                turn_text = ""

            return TranscriptionSegment(
                speaker=turn.speaker,
                start=turn.start,
                end=turn.end,
                duration=turn.duration,
                text=turn_text,
                turn_id=turn_id,
            )

        except Exception as e:
            logger.error(f"Failed to transcribe turn {turn_id}: {e}")
            return TranscriptionSegment(
                speaker=turn.speaker,
                start=turn.start,
                end=turn.end,
                duration=turn.duration,
                text="[Transcription failed]",
                turn_id=turn_id,
                error=str(e),
            )

    def _cleanup_file(self, path: str) -> None:
        """Clean up a single temp file."""
        try:
            if path in self._temp_files:
                self._temp_files.remove(path)
            if os.path.exists(path):
                os.unlink(path)
                logger.debug(f"Cleaned up temp file: {path}")
        except Exception as e:
            logger.warning(f"Failed to clean up temp file {path}: {e}")

    def _cleanup_all_temp_files(self) -> None:
        """Clean up all tracked temp files."""
        for path in list(self._temp_files):
            self._cleanup_file(path)
        self._temp_files.clear()


def free_gpu_memory() -> None:
    """Free GPU memory by running garbage collection and clearing CUDA cache.

    Call this between heavy operations (e.g., after unloading diarization
    model and before loading ASR model) to prevent memory issues.
    """
    gc.collect()

    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            logger.info("Cleared CUDA cache and synchronized")
    except ImportError:
        pass


def build_full_transcript(segments: List[TranscriptionSegment]) -> str:
    """Build a full transcript from transcription segments.

    Args:
        segments: List of transcription segments

    Returns:
        Full transcript with speaker attribution
    """
    lines = []
    for seg in segments:
        lines.append(f"[{seg.speaker}] {seg.text}")
    return "\n".join(lines)


def create_transcription_result(
    segments: List[TranscriptionSegment],
    diarization_result: DiarizationResult,
) -> TranscriptionResult:
    """Create a complete TranscriptionResult from segments and diarization info.

    Args:
        segments: List of transcription segments
        diarization_result: Original diarization result

    Returns:
        Complete TranscriptionResult
    """
    successful = sum(1 for seg in segments if seg.error is None)
    failed = sum(1 for seg in segments if seg.error is not None)

    return TranscriptionResult(
        segments=segments,
        full_transcript=build_full_transcript(segments),
        num_speakers=diarization_result.num_speakers,
        total_duration=diarization_result.total_duration,
        speakers=diarization_result.speakers,
        successful_turns=successful,
        failed_turns=failed,
    )

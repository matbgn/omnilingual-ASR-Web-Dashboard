"""Diarization models for speaker attribution in ASR."""

from .pyannote_diarizer import PyannoteDiarizer, PYANNOTE_AVAILABLE
from .config import DiarizationConfig
from .service import DiarizationService
from .diarization_types import (
    DiarizationResult,
    SpeakerSegment,
    DiarizationProgress,
    DiarizationStatus,
    ProgressCallback,
)
from .chunking import SmartChunkBuilder, SpeakerTurn
from .transcription_pipeline import (
    SpeakerTurnTranscriptionPipeline,
    TranscriptionSegment,
    TranscriptionProgress,
    TranscriptionResult,
    free_gpu_memory,
    build_full_transcript,
    create_transcription_result,
)
from .formatting import (
    DiarizedTranscriptionResult,
    EXPORT_FORMATS,
)

__all__ = [
    "PyannoteDiarizer",
    "DiarizationConfig",
    "DiarizationService",
    "DiarizationResult",
    "SpeakerSegment",
    "DiarizationProgress",
    "DiarizationStatus",
    "ProgressCallback",
    "SmartChunkBuilder",
    "SpeakerTurn",
    "PYANNOTE_AVAILABLE",
    # Transcription pipeline
    "SpeakerTurnTranscriptionPipeline",
    "TranscriptionSegment",
    "TranscriptionProgress",
    "TranscriptionResult",
    "free_gpu_memory",
    "build_full_transcript",
    "create_transcription_result",
    # Formatting and export
    "DiarizedTranscriptionResult",
    "EXPORT_FORMATS",
]

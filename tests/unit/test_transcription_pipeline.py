# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for speaker turn transcription pipeline."""

import pytest
import os
import sys
from unittest.mock import Mock, MagicMock, patch
import tempfile

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))

from omnilingual_asr.models.diarization.transcription_pipeline import (
    TranscriptionSegment,
    TranscriptionProgress,
    TranscriptionResult,
    SpeakerTurnTranscriptionPipeline,
    free_gpu_memory,
    build_full_transcript,
    create_transcription_result,
)
from omnilingual_asr.models.diarization.chunking import SpeakerTurn
from omnilingual_asr.models.diarization.diarization_types import (
    SpeakerSegment,
    DiarizationResult,
)


def test_transcription_segment_creation():
    """Test TranscriptionSegment creation and validation."""
    segment = TranscriptionSegment(
        speaker="SPEAKER_00",
        start=0.0,
        end=2.5,
        duration=2.5,
        text="Hello world",
        turn_id=0,
    )

    assert segment.speaker == "SPEAKER_00"
    assert segment.start == 0.0
    assert segment.end == 2.5
    assert segment.duration == 2.5
    assert segment.text == "Hello world"
    assert segment.turn_id == 0
    assert segment.error is None


def test_transcription_segment_with_error():
    """Test TranscriptionSegment with error."""
    segment = TranscriptionSegment(
        speaker="SPEAKER_00",
        start=0.0,
        end=2.5,
        duration=2.5,
        text="[Transcription failed]",
        turn_id=0,
        error="Model error",
    )

    assert segment.error == "Model error"
    seg_dict = segment.to_dict()
    assert "error" in seg_dict
    assert seg_dict["error"] == "Model error"


def test_transcription_segment_to_dict():
    """Test TranscriptionSegment conversion to dict."""
    segment = TranscriptionSegment(
        speaker="SPEAKER_01",
        start=5.0,
        end=10.0,
        duration=5.0,
        text="Test transcript",
        turn_id=1,
    )

    seg_dict = segment.to_dict()
    assert seg_dict["speaker"] == "SPEAKER_01"
    assert seg_dict["start"] == 5.0
    assert seg_dict["end"] == 10.0
    assert seg_dict["duration"] == 5.0
    assert seg_dict["text"] == "Test transcript"
    assert seg_dict["turn_id"] == 1
    assert "error" not in seg_dict


def test_transcription_segment_from_dict():
    """Test TranscriptionSegment creation from dict."""
    data = {
        "speaker": "SPEAKER_00",
        "start": 1.0,
        "end": 3.0,
        "duration": 2.0,
        "text": "From dict",
        "turn_id": 2,
    }

    segment = TranscriptionSegment.from_dict(data)
    assert segment.speaker == "SPEAKER_00"
    assert segment.start == 1.0
    assert segment.end == 3.0
    assert segment.text == "From dict"


def test_transcription_progress_creation():
    """Test TranscriptionProgress creation."""
    progress = TranscriptionProgress(
        stage="transcribing",
        current_turn=3,
        total_turns=10,
        speaker="SPEAKER_00",
        message="Processing turn 3/10",
    )

    assert progress.stage == "transcribing"
    assert progress.current_turn == 3
    assert progress.total_turns == 10
    assert progress.speaker == "SPEAKER_00"
    assert progress.message == "Processing turn 3/10"


def test_transcription_progress_percent():
    """Test progress percentage calculation."""
    progress = TranscriptionProgress(
        stage="transcribing",
        current_turn=5,
        total_turns=10,
    )
    assert progress.progress_percent == 50

    progress_zero = TranscriptionProgress(
        stage="transcribing",
        current_turn=0,
        total_turns=0,
    )
    assert progress_zero.progress_percent == 0


def test_transcription_progress_to_dict():
    """Test TranscriptionProgress conversion to dict."""
    progress = TranscriptionProgress(
        stage="extracting",
        current_turn=2,
        total_turns=5,
        speaker="SPEAKER_01",
        message="Extracting audio",
    )

    prog_dict = progress.to_dict()
    assert prog_dict["stage"] == "extracting"
    assert prog_dict["progress"] == 40  # 2/5 = 40%
    assert prog_dict["current_turn"] == 2
    assert prog_dict["total_turns"] == 5
    assert prog_dict["speaker"] == "SPEAKER_01"


def test_transcription_result_creation():
    """Test TranscriptionResult creation."""
    segments = [
        TranscriptionSegment(
            speaker="SPEAKER_00",
            start=0.0,
            end=2.5,
            duration=2.5,
            text="Hello",
            turn_id=0,
        ),
        TranscriptionSegment(
            speaker="SPEAKER_01",
            start=3.0,
            end=5.0,
            duration=2.0,
            text="Hi there",
            turn_id=1,
        ),
    ]

    result = TranscriptionResult(
        segments=segments,
        full_transcript="[SPEAKER_00] Hello\n[SPEAKER_01] Hi there",
        num_speakers=2,
        total_duration=5.0,
        speakers=["SPEAKER_00", "SPEAKER_01"],
        successful_turns=2,
        failed_turns=0,
    )

    assert len(result.segments) == 2
    assert result.num_speakers == 2
    assert result.successful_turns == 2
    assert result.failed_turns == 0


def test_transcription_result_to_dict():
    """Test TranscriptionResult conversion to dict."""
    segments = [
        TranscriptionSegment(
            speaker="SPEAKER_00",
            start=0.0,
            end=2.5,
            duration=2.5,
            text="Test",
            turn_id=0,
        ),
    ]

    result = TranscriptionResult(
        segments=segments,
        full_transcript="[SPEAKER_00] Test",
        num_speakers=1,
        total_duration=2.5,
        speakers=["SPEAKER_00"],
    )

    result_dict = result.to_dict()
    assert "segments" in result_dict
    assert "full_transcript" in result_dict
    assert "num_speakers" in result_dict
    assert "total_duration" in result_dict
    assert "successful_turns" in result_dict
    assert "failed_turns" in result_dict


def test_build_full_transcript():
    """Test building full transcript from segments."""
    segments = [
        TranscriptionSegment(
            speaker="SPEAKER_00",
            start=0.0,
            end=2.0,
            duration=2.0,
            text="First line",
            turn_id=0,
        ),
        TranscriptionSegment(
            speaker="SPEAKER_01",
            start=2.5,
            end=5.0,
            duration=2.5,
            text="Second line",
            turn_id=1,
        ),
        TranscriptionSegment(
            speaker="SPEAKER_00",
            start=5.5,
            end=7.0,
            duration=1.5,
            text="Third line",
            turn_id=2,
        ),
    ]

    transcript = build_full_transcript(segments)
    expected = "[SPEAKER_00] First line\n[SPEAKER_01] Second line\n[SPEAKER_00] Third line"
    assert transcript == expected


def test_create_transcription_result():
    """Test creating TranscriptionResult from segments and diarization result."""
    segments = [
        TranscriptionSegment(
            speaker="SPEAKER_00",
            start=0.0,
            end=2.0,
            duration=2.0,
            text="Success",
            turn_id=0,
        ),
        TranscriptionSegment(
            speaker="SPEAKER_01",
            start=3.0,
            end=5.0,
            duration=2.0,
            text="[Transcription failed]",
            turn_id=1,
            error="Test error",
        ),
    ]

    diar_result = DiarizationResult(
        segments=[
            SpeakerSegment(start=0.0, end=2.0, speaker="SPEAKER_00"),
            SpeakerSegment(start=3.0, end=5.0, speaker="SPEAKER_01"),
        ],
        speakers=["SPEAKER_00", "SPEAKER_01"],
        num_speakers=2,
        total_duration=5.0,
    )

    result = create_transcription_result(segments, diar_result)
    assert result.successful_turns == 1
    assert result.failed_turns == 1
    assert result.num_speakers == 2


def test_free_gpu_memory():
    """Test GPU memory cleanup function."""
    # Should not raise even without GPU
    free_gpu_memory()


class TestSpeakerTurnTranscriptionPipeline:
    """Tests for SpeakerTurnTranscriptionPipeline."""

    def test_pipeline_initialization(self):
        """Test pipeline initialization."""
        mock_asr = Mock()
        pipeline = SpeakerTurnTranscriptionPipeline(
            asr_pipeline=mock_asr,
            batch_size=1,
            cleanup_temp_files=True,
        )

        assert pipeline.asr_pipeline == mock_asr
        assert pipeline.batch_size == 1
        assert pipeline.cleanup_temp_files is True
        assert pipeline.chunk_builder is not None

    def test_transcribe_single_turn(self):
        """Test single turn transcription."""
        mock_asr = Mock()
        mock_asr.transcribe.return_value = ["Hello world"]

        pipeline = SpeakerTurnTranscriptionPipeline(asr_pipeline=mock_asr)

        # Create a mock turn
        turn = SpeakerTurn(
            start=0.0,
            end=2.0,
            speaker="SPEAKER_00",
            original_segments=[SpeakerSegment(start=0.0, end=2.0, speaker="SPEAKER_00")],
        )

        result = pipeline._transcribe_single_turn(
            "/tmp/test.wav", turn, 0, "en"
        )

        assert result.speaker == "SPEAKER_00"
        assert result.text == "Hello world"
        assert result.turn_id == 0
        assert result.error is None

    def test_transcribe_single_turn_failure(self):
        """Test single turn transcription failure handling."""
        mock_asr = Mock()
        mock_asr.transcribe.side_effect = Exception("ASR failed")

        pipeline = SpeakerTurnTranscriptionPipeline(asr_pipeline=mock_asr)

        turn = SpeakerTurn(
            start=0.0,
            end=2.0,
            speaker="SPEAKER_00",
            original_segments=[SpeakerSegment(start=0.0, end=2.0, speaker="SPEAKER_00")],
        )

        result = pipeline._transcribe_single_turn(
            "/tmp/test.wav", turn, 0, "en"
        )

        assert result.text == "[Transcription failed]"
        assert result.error == "ASR failed"

    def test_transcribe_turns_with_callbacks(self):
        """Test transcription with progress and segment callbacks."""
        mock_asr = Mock()
        mock_asr.transcribe.return_value = ["Transcribed text"]

        # Mock chunk builder to avoid actual audio extraction
        mock_chunk_builder = Mock()
        mock_chunk_builder.extract_turn_audio.return_value = "/tmp/mock_turn.wav"

        pipeline = SpeakerTurnTranscriptionPipeline(
            asr_pipeline=mock_asr,
            chunk_builder=mock_chunk_builder,
            cleanup_temp_files=True,
        )

        turns = [
            SpeakerTurn(
                start=0.0,
                end=2.0,
                speaker="SPEAKER_00",
                original_segments=[SpeakerSegment(start=0.0, end=2.0, speaker="SPEAKER_00")],
            ),
        ]

        progress_updates = []
        segments_received = []

        def progress_cb(progress):
            progress_updates.append(progress)

        def segment_cb(segment):
            segments_received.append(segment)

        with patch("os.path.exists", return_value=True):
            with patch("os.unlink"):
                results = pipeline.transcribe_turns(
                    "/tmp/test.wav",
                    turns,
                    "en",
                    progress_callback=progress_cb,
                    segment_callback=segment_cb,
                )

        assert len(results) == 1
        assert results[0].text == "Transcribed text"
        assert len(segments_received) == 1
        # Should have extracting, transcribing, and complete progress updates
        assert len(progress_updates) >= 2

    def test_transcribe_turns_streaming(self):
        """Test streaming transcription."""
        mock_asr = Mock()
        mock_asr.transcribe.return_value = ["Stream text"]

        mock_chunk_builder = Mock()
        mock_chunk_builder.extract_turn_audio.return_value = "/tmp/mock_turn.wav"

        pipeline = SpeakerTurnTranscriptionPipeline(
            asr_pipeline=mock_asr,
            chunk_builder=mock_chunk_builder,
        )

        turns = [
            SpeakerTurn(
                start=0.0,
                end=2.0,
                speaker="SPEAKER_00",
                original_segments=[SpeakerSegment(start=0.0, end=2.0, speaker="SPEAKER_00")],
            ),
        ]

        events = []
        with patch("os.path.exists", return_value=True):
            with patch("os.unlink"):
                for event in pipeline.transcribe_turns_streaming(
                    "/tmp/test.wav", turns, "en"
                ):
                    events.append(event)

        # Should get progress events and segment events
        progress_events = [e for e in events if isinstance(e, TranscriptionProgress)]
        segment_events = [e for e in events if isinstance(e, TranscriptionSegment)]

        assert len(segment_events) == 1
        assert segment_events[0].text == "Stream text"
        # Check for complete event
        complete_events = [e for e in progress_events if e.stage == "complete"]
        assert len(complete_events) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

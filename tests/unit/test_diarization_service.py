# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for diarization service."""

import pytest
import tempfile
import os
import sys

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))

from omnilingual_asr.models.diarization.service import DiarizationService
from omnilingual_asr.models.diarization.config import DiarizationConfig
from omnilingual_asr.models.diarization.diarization_types import (
    SpeakerSegment,
    DiarizationResult,
    DiarizationProgress,
)


def test_diarization_service_initialization():
    """Test basic DiarizationService initialization."""
    config = DiarizationConfig(
        model_name="pyannote/speaker-diarization-3.1",
        device="cpu",
        min_speakers=1,
        max_speakers=2,
    )

    service = DiarizationService(config)
    assert service.config == config
    assert not service._is_loaded
    assert service.diarizer is None
    assert service._hf_token is None


def test_diarization_service_status():
    """Test service status reporting."""
    service = DiarizationService()

    status = service.get_status()
    assert "pyannote_available" in status
    assert "hf_token_set" in status
    assert "model_loaded" in status
    assert "device" in status
    assert "model_name" in status

    # Initially should not have token set
    assert not status["hf_token_set"]
    assert not status["model_loaded"]


def test_diarization_service_token_management():
    """Test HF token management."""
    service = DiarizationService()

    # Initially no token
    assert not service._hf_token

    # Set token
    fake_token = "fake_token_for_testing"
    service.set_hf_token(fake_token)
    assert service._hf_token == fake_token

    # Check status reflects token
    status = service.get_status()
    assert status["hf_token_set"]


def test_speaker_segment_creation():
    """Test SpeakerSegment creation and validation."""
    # Valid segment
    segment = SpeakerSegment(start=0.0, end=2.5, speaker="SPEAKER_00")
    assert segment.start == 0.0
    assert segment.end == 2.5
    assert segment.speaker == "SPEAKER_00"
    assert segment.duration == 2.5

    # Test conversion to dict
    seg_dict = segment.to_dict()
    assert seg_dict["start"] == 0.0
    assert seg_dict["end"] == 2.5
    assert seg_dict["speaker"] == "SPEAKER_00"
    assert seg_dict["duration"] == 2.5

    # Test creation from dict
    segment2 = SpeakerSegment.from_dict(seg_dict)
    assert segment2.start == segment.start
    assert segment2.end == segment.end
    assert segment2.speaker == segment.speaker

    # Test validation
    with pytest.raises(ValueError, match="Start time cannot be negative"):
        SpeakerSegment(start=-1.0, end=2.5, speaker="SPEAKER_00")

    with pytest.raises(ValueError, match="End time must be greater than start time"):
        SpeakerSegment(start=2.5, end=2.5, speaker="SPEAKER_00")

    with pytest.raises(ValueError, match="Speaker identifier cannot be empty"):
        SpeakerSegment(start=0.0, end=2.5, speaker="")


def test_diarization_result_creation():
    """Test DiarizationResult creation and validation."""
    segments = [
        SpeakerSegment(start=0.0, end=2.5, speaker="SPEAKER_00"),
        SpeakerSegment(start=2.7, end=5.0, speaker="SPEAKER_01"),
    ]

    result = DiarizationResult(
        segments=segments,
        speakers=["SPEAKER_00", "SPEAKER_01"],
        num_speakers=2,
        total_duration=5.0,
    )

    assert len(result.segments) == 2
    assert result.num_speakers == 2
    assert result.total_duration == 5.0
    assert result.speakers == ["SPEAKER_00", "SPEAKER_01"]

    # Test getting segments by speaker
    speaker_00_segments = result.get_segments_by_speaker("SPEAKER_00")
    assert len(speaker_00_segments) == 1
    assert speaker_00_segments[0].speaker == "SPEAKER_00"

    # Test getting speaker total time
    speaker_00_time = result.get_speaker_total_time("SPEAKER_00")
    assert speaker_00_time == 2.5

    # Test conversion to dict
    result_dict = result.to_dict()
    assert "segments" in result_dict
    assert "speakers" in result_dict
    assert "num_speakers" in result_dict
    assert "total_duration" in result_dict

    # Test creation from dict
    result2 = DiarizationResult.from_dict(result_dict)
    assert result2.num_speakers == result.num_speakers
    assert len(result2.segments) == len(result.segments)


def test_diarization_progress():
    """Test DiarizationProgress creation."""
    progress = DiarizationProgress(
        stage="diarizing",
        progress=0.5,
        message="Processing...",
        current_step=1,
        total_steps=2,
    )

    assert progress.stage == "diarizing"
    assert progress.progress == 0.5
    assert progress.message == "Processing..."
    assert progress.current_step == 1
    assert progress.total_steps == 2

    # Test conversion to dict
    progress_dict = progress.to_dict()
    assert progress_dict["stage"] == "diarizing"
    assert progress_dict["progress"] == 0.5


def test_audio_validation_without_model():
    """Test audio file validation when model is not loaded."""
    service = DiarizationService()

    # Test with non-existent file
    validation = service.validate_audio_file("/nonexistent/file.wav")
    assert not validation["valid"]
    assert "File not found" in validation["error"]

    # Test with model not loaded
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        validation = service.validate_audio_file(tmp_path)
        assert not validation["valid"]
        assert "not loaded" in validation["error"]
    finally:
        os.unlink(tmp_path)


def test_model_loading_without_token():
    """Test that model loading fails without token."""
    service = DiarizationService()

    with pytest.raises(ValueError, match="Hugging Face token is required"):
        service.load_model()

    with pytest.raises(ValueError, match="Model not loaded"):
        service.diarize("test.wav")


def test_model_unloading():
    """Test model unloading."""
    service = DiarizationService()

    # Should not fail even when no model is loaded
    service.unload_model()
    assert not service._is_loaded
    assert service.diarizer is None


def test_context_manager():
    """Test using DiarizationService as context manager."""
    config = DiarizationConfig(device="cpu")

    with DiarizationService(config) as service:
        assert service.config == config
        assert not service._is_loaded

    # Model should be unloaded after context exit
    assert not service._is_loaded


def test_get_model_info_without_model():
    """Test getting model info when no model is loaded."""
    service = DiarizationService()

    info = service.get_model_info()
    assert info["loaded"] is False
    assert "config" not in info


def test_is_availability():
    """Test availability checking."""
    service = DiarizationService()

    # Without token should not be available
    assert not service.is_available()

    # With token might be available depending on pyannote installation
    service.set_hf_token("fake_token")
    availability = service.is_available()
    # This depends on whether pyannote is installed, so we just check it returns a boolean
    assert isinstance(availability, bool)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

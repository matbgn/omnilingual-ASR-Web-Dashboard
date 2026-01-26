"""Unit tests for the formatting module."""

import json
import pytest
from src.omnilingual_asr.models.diarization.formatting import (
    DiarizedTranscriptionResult,
    EXPORT_FORMATS,
    _format_timestamp,
    _format_time_short,
)
from src.omnilingual_asr.models.diarization.transcription_pipeline import (
    TranscriptionSegment,
)


class TestTimestampFormatting:
    """Tests for timestamp formatting functions."""

    def test_format_timestamp_zero(self):
        """Test formatting zero seconds."""
        assert _format_timestamp(0.0) == "00:00:00,000"

    def test_format_timestamp_seconds(self):
        """Test formatting seconds only."""
        assert _format_timestamp(45.5) == "00:00:45,500"

    def test_format_timestamp_minutes(self):
        """Test formatting minutes and seconds."""
        assert _format_timestamp(125.25) == "00:02:05,250"

    def test_format_timestamp_hours(self):
        """Test formatting hours, minutes, and seconds."""
        assert _format_timestamp(3661.123) == "01:01:01,123"

    def test_format_time_short_seconds(self):
        """Test short format for seconds only."""
        assert _format_time_short(45.5) == "00:45"

    def test_format_time_short_minutes(self):
        """Test short format for minutes and seconds."""
        assert _format_time_short(125.25) == "02:05"

    def test_format_time_short_hours(self):
        """Test short format includes hours when needed."""
        assert _format_time_short(3661.5) == "01:01:01"


class TestDiarizedTranscriptionResult:
    """Tests for DiarizedTranscriptionResult class."""

    @pytest.fixture
    def sample_segments(self):
        """Create sample transcription segments."""
        return [
            TranscriptionSegment(
                speaker="Speaker_00",
                start=0.0,
                end=5.5,
                duration=5.5,
                text="Hello, how are you?",
                turn_id=0,
            ),
            TranscriptionSegment(
                speaker="Speaker_01",
                start=6.0,
                end=12.0,
                duration=6.0,
                text="I'm doing great, thanks for asking!",
                turn_id=1,
            ),
            TranscriptionSegment(
                speaker="Speaker_00",
                start=12.5,
                end=18.0,
                duration=5.5,
                text="That's wonderful to hear.",
                turn_id=2,
            ),
        ]

    @pytest.fixture
    def sample_result(self, sample_segments):
        """Create a sample DiarizedTranscriptionResult."""
        return DiarizedTranscriptionResult(
            segments=sample_segments,
            full_transcript="[Speaker_00] Hello, how are you?\n[Speaker_01] I'm doing great, thanks for asking!\n[Speaker_00] That's wonderful to hear.",
            num_speakers=2,
            total_duration=18.0,
            speakers=["Speaker_00", "Speaker_01"],
            successful_turns=3,
            failed_turns=0,
            filename="test_audio.wav",
            lang_code="eng_Latn",
            timestamp="2025-01-25T10:30:00",
        )

    @pytest.fixture
    def sample_history_entry(self):
        """Create a sample history entry."""
        return {
            "id": 5,
            "filename": "meeting_recording.wav",
            "lang_code": "eng_Latn",
            "transcription": "[Speaker_00] Hello\n[Speaker_01] Hi",
            "diarization": {
                "segments": [
                    {
                        "speaker": "Speaker_00",
                        "start": 0.0,
                        "end": 2.5,
                        "duration": 2.5,
                        "text": "Hello",
                        "turn_id": 0,
                    },
                    {
                        "speaker": "Speaker_01",
                        "start": 3.0,
                        "end": 5.0,
                        "duration": 2.0,
                        "text": "Hi",
                        "turn_id": 1,
                    },
                ],
                "full_transcript": "[Speaker_00] Hello\n[Speaker_01] Hi",
                "num_speakers": 2,
                "total_duration": 5.0,
                "speakers": ["Speaker_00", "Speaker_01"],
                "successful_turns": 2,
                "failed_turns": 0,
            },
            "timestamp": "2025-01-25T10:00:00",
        }

    def test_from_history_entry(self, sample_history_entry):
        """Test creating result from history entry."""
        result = DiarizedTranscriptionResult.from_history_entry(sample_history_entry)

        assert result.num_speakers == 2
        assert len(result.segments) == 2
        assert result.filename == "meeting_recording.wav"
        assert result.segments[0].speaker == "Speaker_00"
        assert result.segments[0].text == "Hello"

    def test_from_history_entry_missing_diarization(self):
        """Test error when history entry has no diarization data."""
        entry = {"id": 1, "transcription": "Hello"}
        with pytest.raises(ValueError, match="does not contain diarization data"):
            DiarizedTranscriptionResult.from_history_entry(entry)

    def test_to_inline(self, sample_result):
        """Test inline format export."""
        result = sample_result.to_inline()

        assert "[Speaker_00] Hello, how are you?" in result
        assert "[Speaker_01] I'm doing great, thanks for asking!" in result
        assert "[Speaker_00] That's wonderful to hear." in result
        # Check line count
        lines = result.strip().split("\n")
        assert len(lines) == 3

    def test_to_timestamped(self, sample_result):
        """Test timestamped format export."""
        result = sample_result.to_timestamped()

        # Check timestamp format presence
        assert "[00:00 - 00:05] Speaker_00:" in result
        assert "[00:06 - 00:12] Speaker_01:" in result
        assert "[00:12 - 00:18] Speaker_00:" in result
        # Check text is present
        assert "Hello, how are you?" in result

    def test_to_json(self, sample_result):
        """Test JSON format export."""
        result = sample_result.to_json()

        # Parse and validate JSON structure
        data = json.loads(result)
        assert "metadata" in data
        assert "segments" in data
        assert "full_transcript" in data

        # Check metadata
        assert data["metadata"]["filename"] == "test_audio.wav"
        assert data["metadata"]["num_speakers"] == 2

        # Check segments
        assert len(data["segments"]) == 3
        assert data["segments"][0]["speaker"] == "Speaker_00"
        assert data["segments"][0]["text"] == "Hello, how are you?"

    def test_to_srt(self, sample_result):
        """Test SRT format export."""
        result = sample_result.to_srt()

        # Check SRT structure
        lines = result.split("\n")

        # First subtitle
        assert "1" in lines[0]
        assert "00:00:00,000 --> 00:00:05,500" in lines[1]
        assert "[Speaker_00] Hello, how are you?" in lines[2]

        # Second subtitle
        assert "2" in result
        assert "00:00:06,000 --> 00:00:12,000" in result
        assert "[Speaker_01] I'm doing great, thanks for asking!" in result

    def test_export_inline(self, sample_result):
        """Test export method with inline format."""
        result = sample_result.export("inline")
        assert result == sample_result.to_inline()

    def test_export_timestamped(self, sample_result):
        """Test export method with timestamped format."""
        result = sample_result.export("timestamped")
        assert result == sample_result.to_timestamped()

    def test_export_json(self, sample_result):
        """Test export method with json format."""
        result = sample_result.export("json")
        assert result == sample_result.to_json()

    def test_export_srt(self, sample_result):
        """Test export method with srt format."""
        result = sample_result.export("srt")
        assert result == sample_result.to_srt()

    def test_export_invalid_format(self, sample_result):
        """Test export method with invalid format."""
        with pytest.raises(ValueError, match="Unsupported format"):
            sample_result.export("invalid_format")

    def test_get_content_type(self):
        """Test content type lookup."""
        assert DiarizedTranscriptionResult.get_content_type("inline") == "text/plain; charset=utf-8"
        assert DiarizedTranscriptionResult.get_content_type("json") == "application/json; charset=utf-8"
        assert DiarizedTranscriptionResult.get_content_type("srt") == "text/srt; charset=utf-8"
        assert DiarizedTranscriptionResult.get_content_type("unknown") == "text/plain; charset=utf-8"

    def test_get_file_extension(self):
        """Test file extension lookup."""
        assert DiarizedTranscriptionResult.get_file_extension("inline") == ".txt"
        assert DiarizedTranscriptionResult.get_file_extension("timestamped") == ".txt"
        assert DiarizedTranscriptionResult.get_file_extension("json") == ".json"
        assert DiarizedTranscriptionResult.get_file_extension("srt") == ".srt"
        assert DiarizedTranscriptionResult.get_file_extension("unknown") == ".txt"

    def test_to_dict(self, sample_result):
        """Test to_dict conversion."""
        data = sample_result.to_dict()

        assert data["num_speakers"] == 2
        assert data["total_duration"] == 18.0
        assert len(data["segments"]) == 3
        assert data["filename"] == "test_audio.wav"


class TestExportFormats:
    """Tests for EXPORT_FORMATS constant."""

    def test_export_formats_contains_required(self):
        """Test that all required formats are present."""
        assert "inline" in EXPORT_FORMATS
        assert "timestamped" in EXPORT_FORMATS
        assert "json" in EXPORT_FORMATS
        assert "srt" in EXPORT_FORMATS

    def test_export_formats_count(self):
        """Test that we have exactly 4 formats."""
        assert len(EXPORT_FORMATS) == 4


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_segment_with_error(self):
        """Test handling segment with transcription error."""
        segment = TranscriptionSegment(
            speaker="Speaker_00",
            start=0.0,
            end=5.0,
            duration=5.0,
            text="[Transcription failed]",
            turn_id=0,
            error="Audio extraction failed",
        )
        result = DiarizedTranscriptionResult(
            segments=[segment],
            full_transcript="[Speaker_00] [Transcription failed]",
            num_speakers=1,
            total_duration=5.0,
            speakers=["Speaker_00"],
            successful_turns=0,
            failed_turns=1,
        )

        # Test JSON export includes error
        json_output = json.loads(result.to_json())
        assert json_output["segments"][0]["error"] == "Audio extraction failed"

    def test_empty_speakers(self):
        """Test result with empty speakers list but valid segments."""
        segment = TranscriptionSegment(
            speaker="Speaker_00",
            start=0.0,
            end=5.0,
            duration=5.0,
            text="Hello",
            turn_id=0,
        )
        result = DiarizedTranscriptionResult(
            segments=[segment],
            full_transcript="[Speaker_00] Hello",
            num_speakers=1,
            total_duration=5.0,
            speakers=["Speaker_00"],
        )
        # Should work without error
        inline = result.to_inline()
        assert "[Speaker_00] Hello" in inline

    def test_long_duration_timestamps(self):
        """Test formatting for very long audio (hours)."""
        segment = TranscriptionSegment(
            speaker="Speaker_00",
            start=7200.0,  # 2 hours
            end=7260.0,  # 2 hours 1 minute
            duration=60.0,
            text="Long recording segment",
            turn_id=0,
        )
        result = DiarizedTranscriptionResult(
            segments=[segment],
            full_transcript="[Speaker_00] Long recording segment",
            num_speakers=1,
            total_duration=7260.0,
            speakers=["Speaker_00"],
        )

        srt = result.to_srt()
        assert "02:00:00,000 --> 02:01:00,000" in srt

        timestamped = result.to_timestamped()
        assert "02:00:00 - 02:01:00" in timestamped

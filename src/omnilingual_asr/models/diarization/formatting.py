"""Result formatting and export options for diarized transcriptions."""

import json
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

from .transcription_pipeline import TranscriptionSegment


def _format_timestamp(seconds: float) -> str:
    """Format seconds as HH:MM:SS,mmm for SRT format.

    Args:
        seconds: Time in seconds

    Returns:
        Formatted timestamp string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _format_time_short(seconds: float) -> str:
    """Format seconds as MM:SS or HH:MM:SS for display.

    Args:
        seconds: Time in seconds

    Returns:
        Formatted time string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


@dataclass
class DiarizedTranscriptionResult:
    """Wrapper class for diarized transcription data with export methods.

    This class takes the diarization data stored in history and provides
    multiple export formats: inline text, timestamped text, JSON, and SRT.
    """

    segments: List[TranscriptionSegment]
    full_transcript: str
    num_speakers: int
    total_duration: float
    speakers: List[str]
    successful_turns: int = 0
    failed_turns: int = 0
    filename: Optional[str] = None
    lang_code: Optional[str] = None
    timestamp: Optional[str] = None

    @classmethod
    def from_history_entry(cls, entry: Dict[str, Any]) -> "DiarizedTranscriptionResult":
        """Create from a history entry dictionary.

        Args:
            entry: History entry from the transcription history

        Returns:
            DiarizedTranscriptionResult instance

        Raises:
            ValueError: If entry doesn't contain diarization data
        """
        diarization = entry.get("diarization")
        if not diarization:
            raise ValueError("History entry does not contain diarization data")

        # Convert segment dicts to TranscriptionSegment objects
        segments = []
        for seg_data in diarization.get("segments", []):
            segments.append(TranscriptionSegment.from_dict(seg_data))

        return cls(
            segments=segments,
            full_transcript=diarization.get("full_transcript", ""),
            num_speakers=diarization.get("num_speakers", 0),
            total_duration=diarization.get("total_duration", 0.0),
            speakers=diarization.get("speakers", []),
            successful_turns=diarization.get("successful_turns", 0),
            failed_turns=diarization.get("failed_turns", 0),
            filename=entry.get("filename"),
            lang_code=entry.get("lang_code"),
            timestamp=entry.get("timestamp"),
        )

    def to_inline(self) -> str:
        """Export as inline text with speaker labels.

        Format: [Speaker] Text on same line, speakers separated by newlines.

        Returns:
            Inline formatted transcript
        """
        lines = []
        for segment in self.segments:
            lines.append(f"[{segment.speaker}] {segment.text}")
        return "\n".join(lines)

    def to_timestamped(self) -> str:
        """Export as timestamped text with speaker labels and times.

        Format:
        [00:00 - 00:15] Speaker_00:
        Text of the segment here.

        Returns:
            Timestamped formatted transcript
        """
        lines = []
        for segment in self.segments:
            start_time = _format_time_short(segment.start)
            end_time = _format_time_short(segment.end)
            lines.append(f"[{start_time} - {end_time}] {segment.speaker}:")
            lines.append(segment.text)
            lines.append("")  # Empty line between segments
        return "\n".join(lines).rstrip()

    def to_json(self, indent: int = 2) -> str:
        """Export as JSON format.

        Args:
            indent: JSON indentation level (default 2)

        Returns:
            JSON formatted string
        """
        data = {
            "metadata": {
                "filename": self.filename,
                "lang_code": self.lang_code,
                "timestamp": self.timestamp,
                "num_speakers": self.num_speakers,
                "total_duration": self.total_duration,
                "speakers": self.speakers,
                "successful_turns": self.successful_turns,
                "failed_turns": self.failed_turns,
            },
            "segments": [
                {
                    "speaker": seg.speaker,
                    "start": seg.start,
                    "end": seg.end,
                    "duration": seg.duration,
                    "text": seg.text,
                    "turn_id": seg.turn_id,
                    "error": seg.error,
                }
                for seg in self.segments
            ],
            "full_transcript": self.full_transcript,
        }
        return json.dumps(data, ensure_ascii=False, indent=indent)

    def to_srt(self) -> str:
        """Export as SRT (SubRip) subtitle format.

        Format:
        1
        00:00:00,000 --> 00:00:15,500
        [Speaker_00] Text of the segment.

        Returns:
            SRT formatted string
        """
        lines = []
        for i, segment in enumerate(self.segments, start=1):
            # Sequence number
            lines.append(str(i))
            # Timestamps
            start_ts = _format_timestamp(segment.start)
            end_ts = _format_timestamp(segment.end)
            lines.append(f"{start_ts} --> {end_ts}")
            # Text with speaker label
            lines.append(f"[{segment.speaker}] {segment.text}")
            # Blank line separator
            lines.append("")
        return "\n".join(lines).rstrip()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format.

        Returns:
            Dictionary representation
        """
        return {
            "segments": [seg.to_dict() for seg in self.segments],
            "full_transcript": self.full_transcript,
            "num_speakers": self.num_speakers,
            "total_duration": self.total_duration,
            "speakers": self.speakers,
            "successful_turns": self.successful_turns,
            "failed_turns": self.failed_turns,
            "filename": self.filename,
            "lang_code": self.lang_code,
            "timestamp": self.timestamp,
        }

    def rename_speakers(self, mapping: Dict[str, str]) -> None:
        """Apply speaker renames to all segments and the speaker list.

        Args:
            mapping: Dictionary mapping original speaker IDs to new labels
        """
        if not mapping:
            return

        # Update individual segments
        for segment in self.segments:
            if segment.speaker in mapping:
                segment.speaker = mapping[segment.speaker]

        # Update the list of all speakers
        self.speakers = [mapping.get(s, s) for s in self.speakers]

        # Update the full transcript summary (if needed, though export usually rebuilds it)
        # We don't really use self.full_transcript for exports, we use segments.

    def export(self, format: str) -> str:
        """Export in the specified format.

        Args:
            format: Export format - one of 'inline', 'timestamped', 'json', 'srt'

        Returns:
            Formatted transcript string

        Raises:
            ValueError: If format is not supported
        """
        format_methods = {
            "inline": self.to_inline,
            "timestamped": self.to_timestamped,
            "json": self.to_json,
            "srt": self.to_srt,
        }

        if format not in format_methods:
            valid_formats = ", ".join(format_methods.keys())
            raise ValueError(f"Unsupported format '{format}'. Valid formats: {valid_formats}")

        return format_methods[format]()

    @staticmethod
    def get_content_type(format: str) -> str:
        """Get the appropriate Content-Type for a format.

        Args:
            format: Export format

        Returns:
            MIME type string
        """
        content_types = {
            "inline": "text/plain; charset=utf-8",
            "timestamped": "text/plain; charset=utf-8",
            "json": "application/json; charset=utf-8",
            "srt": "text/srt; charset=utf-8",
        }
        return content_types.get(format, "text/plain; charset=utf-8")

    @staticmethod
    def get_file_extension(format: str) -> str:
        """Get the appropriate file extension for a format.

        Args:
            format: Export format

        Returns:
            File extension including the dot
        """
        extensions = {
            "inline": ".txt",
            "timestamped": ".txt",
            "json": ".json",
            "srt": ".srt",
        }
        return extensions.get(format, ".txt")


# Supported export formats
EXPORT_FORMATS = ["inline", "timestamped", "json", "srt"]

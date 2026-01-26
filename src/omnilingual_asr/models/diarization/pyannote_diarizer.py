"""Pyannote-based diarization implementation."""

import logging
from typing import List, Optional, Dict, Any, Mapping

import torch

from .config import DiarizationConfig
from .diarization_types import DiarizationProgress

logger = logging.getLogger(__name__)

try:
    from pyannote.audio import Pipeline
    from pyannote.core import Segment, Timeline, Annotation

    PYANNOTE_AVAILABLE = True
except ImportError:
    PYANNOTE_AVAILABLE = False
    Pipeline = None
    Segment = None
    Timeline = None
    Annotation = None


class PyannoteDiarizer:
    """Speaker diarization using pyannote.audio."""

    def __init__(self, config: DiarizationConfig):
        """Initialize the diarizer.

        Args:
            config: Diarization configuration
        """
        if not PYANNOTE_AVAILABLE:
            raise ImportError(
                "pyannote.audio is not installed. Install with: "
                "pip install pyannote.audio"
            )

        self.config = config
        self.pipeline: Optional[Pipeline] = None
        self.device = self._get_device()

    def _get_device(self) -> str:
        """Determine the appropriate device for processing."""
        if self.config.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.config.device

    def load_model(self, auth_token: Optional[str] = None) -> None:
        """Load the diarization pipeline.

        Args:
            auth_token: Hugging Face authentication token for private models
        """
        try:
            self.pipeline = Pipeline.from_pretrained(
                self.config.model_name,
                token=auth_token or self.config.use_auth_token,
            )

            # Move pipeline to specified device
            if hasattr(self.pipeline, "to"):
                self.pipeline.to(torch.device(self.device))

            logger.info(
                f"Loaded diarization pipeline {self.config.model_name} on {self.device}"
            )

        except Exception as e:
            logger.error(f"Failed to load diarization pipeline: {e}")
            raise

    def process(self, audio_path: str, **kwargs) -> Dict[str, Any]:
        """Process an audio file for speaker diarization.

        Args:
            audio_path: Path to audio file
            **kwargs: Additional arguments passed to the pipeline

        Returns:
            Dictionary containing diarization results
        """
        if self.pipeline is None:
            self.load_model()

        try:
            # Define a custom hook to forward progress to our callback
            class CustomProgressHook:
                def __init__(self, callback):
                    self.callback = callback
                    self.steps = {
                        "segmentation": "Segmenting",
                        "embeddings": "Extracting embeddings",
                        "clustering": "Clustering",
                        "speaker_counting": "Counting speakers",
                        "discrete_diarization": "Finalizing",
                    }

                def __call__(
                    self,
                    step_name: str,
                    step_artifact: Any,
                    file: Optional[Mapping] = None,
                    total: Optional[int] = None,
                    completed: Optional[int] = None,
                ):
                    if self.callback:
                        if completed is None or total is None or total == 0:
                            # Indeterminate progress or end of step
                            progress = 1.0
                        else:
                            progress = completed / total
                        
                        readable_step = self.steps.get(step_name, step_name)
                        
                        # We map pyannote's internal steps to our progress roughly
                        # Segmentation is usually the longest part.
                        
                        self.callback(
                            DiarizationProgress(
                                stage="diarizing",
                                progress=progress,
                                message=f"{readable_step}... {progress:.0%}",
                            )
                        )

            # Extract progress callback if provided
            progress_callback = kwargs.get("progress_hook")
            
            # Clean kwargs for the pipeline
            pipeline_kwargs = {
                "min_speakers": self.config.min_speakers,
                "max_speakers": self.config.max_speakers,
            }
            # Add other kwargs except progress_hook
            for k, v in kwargs.items():
                if k != "progress_hook":
                    pipeline_kwargs[k] = v

            # Use our custom hook if callback is available
            hook = CustomProgressHook(progress_callback) if progress_callback else None

            # Run diarization
            diarization = self.pipeline(audio_path, hook=hook, **pipeline_kwargs)

            # Convert to standardized format
            result = self._convert_to_standard_format(diarization)

            logger.info(f"Completed diarization for {audio_path}")
            return result

        except Exception as e:
            logger.error(f"Diarization failed for {audio_path}: {e}")
            raise

    def _convert_to_standard_format(self, diarization: Annotation) -> Dict[str, Any]:
        """Convert pyannote Annotation to standard format.

        Args:
            diarization: pyannote Annotation object

        Returns:
            Standardized diarization result
        """
        segments = []
        speakers = set()

        if hasattr(diarization, "itertracks"):
            # Legacy Annotation object
            iterator = diarization.itertracks(yield_label=True)
        else:
            # Newer DiarizeOutput object or similar iterable
            iterator = diarization.itertracks(yield_label=True) if hasattr(diarization, "itertracks") else diarization.annotation.itertracks(yield_label=True) if hasattr(diarization, "annotation") else ((s, None, l) for s, _, l in diarization.itertracks(yield_label=True)) if hasattr(diarization, "itertracks") else []
            # Wait, the search result says iterate directly over the object? Or use .annotation?
            # Let's try to be robust.
            # search result says: iterate directly yields (turn, speaker)
            # but user code expects (segment, track, speaker)
            pass

        # Let's re-write the loop to be more robust based on search findings
        # Common pyannote pattern is itertracks(yield_label=True) -> (segment, track, label)
        
        if hasattr(diarization, "itertracks"):
            for segment, track, speaker in diarization.itertracks(yield_label=True):
                segments.append({
                    "start": segment.start,
                    "end": segment.end,
                    "duration": segment.duration,
                    "speaker": speaker,
                })
                speakers.add(speaker)
        else:
            # Fallback for DiarizeOutput which wraps Annotation in .speaker_diarization
            source = diarization.speaker_diarization if hasattr(diarization, "speaker_diarization") else diarization
            
            # Check if it has itertracks now (e.g. if it was a wrapper)
            if hasattr(source, "itertracks"):
                for segment, track, speaker in source.itertracks(yield_label=True):
                    segments.append({
                        "start": segment.start,
                        "end": segment.end,
                        "duration": segment.duration,
                        "speaker": speaker,
                    })
                    speakers.add(speaker)
            else:
                # Try direct iteration if it's a newer object type that supports it
                # (segment, label) or (segment, track, label)
                for item in source:
                    if len(item) == 3:
                        segment, _, speaker = item
                    elif len(item) == 2:
                        segment, speaker = item
                    else:
                        continue
                        
                    segments.append({
                        "start": segment.start,
                        "end": segment.end,
                        "duration": segment.duration,
                        "speaker": speaker,
                    })
                    speakers.add(speaker)

        # Sort segments by start time
        segments.sort(key=lambda x: x["start"])

        # Merge consecutive segments from same speaker
        merged_segments = self._merge_consecutive_segments(
            segments, gap_threshold=self.config.segment_merge_threshold
        )

        return {
            "segments": merged_segments,
            "speakers": sorted(list(speakers)),
            "num_speakers": len(speakers),
            "total_duration": max(seg["end"] for seg in segments) if segments else 0,
        }

    def _merge_consecutive_segments(
        self, segments: List[Dict[str, Any]], gap_threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """Merge consecutive segments from the same speaker.

        Args:
            segments: List of speaker segments
            gap_threshold: Maximum gap between segments to merge (seconds)

        Returns:
            List of merged segments
        """
        if not segments:
            return []

        merged = [segments[0].copy()]

        for current in segments[1:]:
            last = merged[-1]

            # Check if segments are from same speaker and close enough
            if (
                last["speaker"] == current["speaker"]
                and current["start"] - last["end"] <= gap_threshold
            ):
                # Merge segments
                last["end"] = current["end"]
                last["duration"] = last["end"] - last["start"]
            else:
                # Add new segment
                merged.append(current.copy())

        return merged

    def get_supported_formats(self) -> List[str]:
        """Get list of supported audio formats."""
        # Common audio formats supported by pyannote/torchaudio
        return [".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".wma", ".aiff", ".au"]

    def __call__(self, audio_path: str, **kwargs) -> Dict[str, Any]:
        """Convenience method to make the diarizer callable.

        Args:
            audio_path: Path to audio file
            **kwargs: Additional arguments

        Returns:
            Diarization results
        """
        return self.process(audio_path, **kwargs)

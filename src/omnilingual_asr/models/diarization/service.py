"""Diarization service for speaker diarization operations."""

import logging
import os
from typing import Optional, Dict, Any, List
import gc

from .diarization_types import (
    DiarizationResult,
    SpeakerSegment,
    DiarizationProgress,
    ProgressCallback,
    DiarizationStatus,
)
from .config import DiarizationConfig
from .pyannote_diarizer import PyannoteDiarizer, PYANNOTE_AVAILABLE

logger = logging.getLogger(__name__)


TOKEN_FILE = ".hf_token"


class DiarizationService:
    """Service for managing speaker diarization operations."""

    def __init__(self, config: Optional[DiarizationConfig] = None):
        """Initialize the diarization service.

        Args:
            config: Diarization configuration. If None, uses defaults.
        """
        self.config = config or DiarizationConfig()
        self.diarizer: Optional[PyannoteDiarizer] = None
        self._is_loaded = False
        self._hf_token: Optional[str] = None
        
        # Load persistent token if available
        if os.path.exists(TOKEN_FILE):
            try:
                with open(TOKEN_FILE, "r") as f:
                    token = f.read().strip()
                    if token:
                        self._hf_token = token
                        logger.info("Loaded persistent Hugging Face token")
            except Exception as e:
                logger.warning(f"Failed to load persistent token: {e}")

    def set_hf_token(self, token: str) -> None:
        """Set the Hugging Face authentication token.

        Args:
            token: Hugging Face authentication token
        """
        self._hf_token = token
        logger.info("Hugging Face token updated")
        
        # Persist token
        try:
            with open(TOKEN_FILE, "w") as f:
                f.write(token)
            logger.info("Hugging Face token saved to disk")
        except Exception as e:
            logger.warning(f"Failed to save persistent token: {e}")

    def is_available(self) -> bool:
        """Check if diarization service is available.

        Returns:
            True if pyannote is available and token is set
        """
        return PYANNOTE_AVAILABLE and bool(self._hf_token)

    def get_status(self) -> Dict[str, Any]:
        """Get current service status.

        Returns:
            Dictionary with service status information
        """
        return {
            "diarization_available": self.is_available(),
            "pyannote_available": PYANNOTE_AVAILABLE,
            "hf_token_set": bool(self._hf_token),
            "model_loaded": self._is_loaded,
            "device": self.config.device if self.config else "auto",
            "model_name": self.config.model_name
            if self.config
            else "pyannote/speaker-diarization-3.1",
        }

    def load_model(self, progress_callback: ProgressCallback = None) -> None:
        """Load the diarization model.

        Args:
            progress_callback: Optional callback for progress updates

        Raises:
            ImportError: If pyannote is not available
            ValueError: If HF token is not set
            RuntimeError: If model loading fails
        """
        if not PYANNOTE_AVAILABLE:
            raise ImportError(
                "pyannote.audio is not installed. Install with: "
                "pip install pyannote.audio"
            )

        if not self._hf_token:
            raise ValueError(
                "Hugging Face token is required. Set it using set_hf_token() method."
            )

        if progress_callback:
            progress_callback(
                DiarizationProgress(
                    stage="loading",
                    progress=0.0,
                    message="Initializing diarization pipeline...",
                )
            )

        try:
            self.diarizer = PyannoteDiarizer(self.config)
            self.diarizer.load_model(auth_token=self._hf_token)
            self._is_loaded = True

            if progress_callback:
                progress_callback(
                    DiarizationProgress(
                        stage="loading",
                        progress=1.0,
                        message="Diarization pipeline loaded successfully",
                    )
                )

            logger.info("Diarization model loaded successfully")

        except Exception as e:
            self._is_loaded = False
            logger.error(f"Failed to load diarization model: {e}")
            raise RuntimeError(f"Failed to load diarization model: {e}") from e

    def unload_model(self) -> None:
        """Unload the diarization model and free GPU memory."""
        if self.diarizer:
            try:
                # Explicitly delete the diarizer
                del self.diarizer
                self.diarizer = None
                self._is_loaded = False

                # Force garbage collection
                gc.collect()

                # Clear CUDA cache if available
                try:
                    import torch

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        logger.info("Cleared CUDA cache")
                except ImportError:
                    pass

                logger.info("Diarization model unloaded and memory freed")

            except Exception as e:
                logger.warning(f"Error during model unloading: {e}")

    def diarize(
        self,
        audio_path: str,
        num_speakers: Optional[int] = None,
        progress_callback: ProgressCallback = None,
    ) -> DiarizationResult:
        """Perform speaker diarization on an audio file.

        Args:
            audio_path: Path to the audio file
            num_speakers: Optional number of speakers (overrides config)
            progress_callback: Optional callback for progress updates

        Returns:
            DiarizationResult with speaker segments

        Raises:
            ValueError: If model is not loaded or audio file doesn't exist
            RuntimeError: If diarization fails
        """
        if not self._is_loaded or not self.diarizer:
            raise ValueError("Model not loaded. Call load_model() first.")

        if not os.path.exists(audio_path):
            raise ValueError(f"Audio file not found: {audio_path}")

        # Check if file format is supported
        supported_formats = self.diarizer.get_supported_formats()
        file_ext = os.path.splitext(audio_path)[1].lower()
        if file_ext not in supported_formats:
            raise ValueError(
                f"Unsupported audio format: {file_ext}. "
                f"Supported formats: {', '.join(supported_formats)}"
            )

        if progress_callback:
            progress_callback(
                DiarizationProgress(
                    stage="diarizing",
                    progress=0.0,
                    message="Starting speaker diarization...",
                )
            )

        try:
            # Prepare kwargs for the diarizer
            kwargs = {}
            if num_speakers:
                kwargs["min_speakers"] = max(1, num_speakers)
                kwargs["max_speakers"] = max(1, num_speakers)

            # Add progress hook to kwargs
            if progress_callback:
                kwargs["progress_hook"] = progress_callback

            # Run diarization
            result_dict = self.diarizer.process(audio_path, **kwargs)

            # Convert to our standardized format
            segments = [
                SpeakerSegment(
                    start=seg["start"],
                    end=seg["end"],
                    speaker=seg["speaker"],
                    confidence=seg.get("confidence"),
                )
                for seg in result_dict["segments"]
            ]

            result = DiarizationResult(
                segments=segments,
                speakers=result_dict["speakers"],
                num_speakers=result_dict["num_speakers"],
                total_duration=result_dict["total_duration"],
            )

            if progress_callback:
                progress_callback(
                    DiarizationProgress(
                        stage="diarizing",
                        progress=1.0,
                        message=f"Diarization complete: {result.num_speakers} speakers found",
                    )
                )

            logger.info(
                f"Diarization completed for {audio_path}: "
                f"{result.num_speakers} speakers, {len(segments)} segments"
            )

            return result

        except Exception as e:
            logger.error(f"Diarization failed for {audio_path}: {e}")

            if progress_callback:
                progress_callback(
                    DiarizationProgress(
                        stage="diarizing",
                        progress=0.0,
                        message=f"Diarization failed: {str(e)}",
                    )
                )

            raise RuntimeError(f"Diarization failed: {e}") from e

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model.

        Returns:
            Dictionary with model information
        """
        if not self._is_loaded or not self.diarizer:
            return {"loaded": False}

        return {
            "loaded": True,
            "model_name": self.config.model_name,
            "device": self.diarizer.device,
            "config": {
                "min_speakers": self.config.min_speakers,
                "max_speakers": self.config.max_speakers,
                "overlap": self.config.overlap,
                "segmentation_duration": self.config.segmentation_duration,
                "clustering_threshold": self.config.clustering_threshold,
            },
        }

    def validate_audio_file(self, audio_path: str) -> Dict[str, Any]:
        """Validate an audio file for diarization.

        Args:
            audio_path: Path to audio file

        Returns:
            Dictionary with validation results
        """
        if not os.path.exists(audio_path):
            return {
                "valid": False,
                "error": "File not found",
                "path": audio_path,
            }

        if not self._is_loaded:
            return {
                "valid": False,
                "error": "Diarization model not loaded",
                "path": audio_path,
            }

        # Check format support
        if self.diarizer:
            supported_formats = self.diarizer.get_supported_formats()
            file_ext = os.path.splitext(audio_path)[1].lower()
            if file_ext not in supported_formats:
                return {
                    "valid": False,
                    "error": f"Unsupported format: {file_ext}",
                    "supported_formats": supported_formats,
                    "path": audio_path,
                }
        else:
            # Fallback check - assume common formats are supported
            supported_formats = [".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"]
            file_ext = os.path.splitext(audio_path)[1].lower()
            if file_ext not in supported_formats:
                return {
                    "valid": False,
                    "error": f"Unsupported format: {file_ext}",
                    "supported_formats": supported_formats,
                    "path": audio_path,
                }

        # Try to get basic audio info
        try:
            # Simple duration check using torchaudio if available
            try:
                import torchaudio

                info = torchaudio.info(audio_path)
                duration = info.num_frames / info.sample_rate
                return {
                    "valid": True,
                    "duration": duration,
                    "format": file_ext,
                    "path": audio_path,
                }
            except ImportError:
                # Fallback - assume valid if file exists and format is supported
                return {
                    "valid": True,
                    "duration": None,
                    "format": file_ext,
                    "path": audio_path,
                    "warning": "Cannot verify duration without torchaudio",
                }
        except Exception as e:
            return {
                "valid": False,
                "error": f"Invalid audio file: {str(e)}",
                "path": audio_path,
            }

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - automatically cleanup."""
        self.unload_model()

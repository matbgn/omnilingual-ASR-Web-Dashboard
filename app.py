#!/usr/bin/env python3
"""
Flask web dashboard for Omnilingual ASR inference.
Provides a simple, beautiful interface for uploading audio files,
transcribing them, and managing transcription history.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import sys
from pathlib import Path
import threading
import queue

# Add src to path so omnilingual_asr can be imported
sys.path.insert(0, str(Path(__file__).parent / "src"))

from datetime import datetime
from pathlib import Path
from typing import List, Optional

from flask import Flask, jsonify, render_template, request, send_file, Response
from werkzeug.utils import secure_filename

from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline
from omnilingual_asr.models.diarization.service import DiarizationService
from omnilingual_asr.models.diarization.chunking import SmartChunkBuilder
from omnilingual_asr.models.diarization.config import DiarizationConfig
from omnilingual_asr.models.diarization.transcription_pipeline import (
    SpeakerTurnTranscriptionPipeline,
    TranscriptionSegment,
    TranscriptionProgress as TxProgress,
    free_gpu_memory,
    build_full_transcript,
)
from omnilingual_asr.models.diarization.formatting import (
    DiarizedTranscriptionResult,
    EXPORT_FORMATS,
)
import random
import uuid
import torch
import shutil
import glob

# --- Dataset Collection Configuration ---
DATASET_DIR = Path("dataset")
RAW_AUDIO_DIR = DATASET_DIR / "raw_audio"
METADATA_FILE = DATASET_DIR / "metadata.jsonl"

# Ensure dataset directories exist
RAW_AUDIO_DIR.mkdir(parents=True, exist_ok=True)


class DatasetManager:
    """Manager for the interactive data collection dataset."""

    PROMPTS = [
        # Bangla Sentences (General conversation, news finish)
        {
            "id": "bn_01",
            "text": "আপনার সাথে দেখা করে খুব ভালো লাগলো।",
            "category": "Bangla",
            "lang_code": "ben_Beng",
        },
        {
            "id": "bn_02",
            "text": "আজকের আবহাওয়া বেশ চমৎকার, তাই না?",
            "category": "Bangla",
            "lang_code": "ben_Beng",
        },
        {
            "id": "bn_03",
            "text": "গতকাল আমি বাজারে গিয়েছিলাম কিছু ফল কিনতে।",
            "category": "Bangla",
            "lang_code": "ben_Beng",
        },
        {
            "id": "bn_04",
            "text": "বাংলাদেশের প্রাকৃতিক সৌন্দর্য আমাকে মুগ্ধ করে।",
            "category": "Bangla",
            "lang_code": "ben_Beng",
        },
        {
            "id": "bn_05",
            "text": "আপনি কি চা খাবেন নাকি কফি?",
            "category": "Bangla",
            "lang_code": "ben_Beng",
        },
        {
            "id": "bn_06",
            "text": "আগামীকাল থেকে পরীক্ষা শুরু হবে।",
            "category": "Bangla",
            "lang_code": "ben_Beng",
        },
        {
            "id": "bn_07",
            "text": "বইটি পড়া শেষ করে আমাকে ফেরত দেবেন।",
            "category": "Bangla",
            "lang_code": "ben_Beng",
        },
        # English Sentences
        {
            "id": "en_01",
            "text": "The quick brown fox jumps over the lazy dog.",
            "category": "English",
            "lang_code": "eng_Latn",
        },
        {
            "id": "en_02",
            "text": "I would like to order a large pizza with extra cheese.",
            "category": "English",
            "lang_code": "eng_Latn",
        },
        {
            "id": "en_03",
            "text": "Can you please tell me the way to the nearest station?",
            "category": "English",
            "lang_code": "eng_Latn",
        },
        {
            "id": "en_04",
            "text": "Technology is changing the way we live and work.",
            "category": "English",
            "lang_code": "eng_Latn",
        },
        {
            "id": "en_05",
            "text": "Please make sure to save your work before closing.",
            "category": "English",
            "lang_code": "eng_Latn",
        },
        # Mixed / Banglish
        {
            "id": "mix_01",
            "text": "আজকের meeting টা খুব important ছিল।",
            "category": "Mixed",
            "lang_code": "ben_Beng",
        },
        {
            "id": "mix_02",
            "text": "আমি কালকে office যাবো না, work from home করবো।",
            "category": "Mixed",
            "lang_code": "ben_Beng",
        },
        {
            "id": "mix_03",
            "text": "প্লিজ file টা আমাকে email করে দিয়েন।",
            "category": "Mixed",
            "lang_code": "ben_Beng",
        },
        {
            "id": "mix_04",
            "text": "Mobile টা charge এ দিয়ে আসো।",
            "category": "Mixed",
            "lang_code": "ben_Beng",
        },
        {
            "id": "mix_05",
            "text": "তোমার presentation টা really awesome হয়েছে।",
            "category": "Mixed",
            "lang_code": "ben_Beng",
        },
        # Custom User Requested (Complex/Mixed)
        {
            "id": "usr_01",
            "text": "Hello Sir, আসসালামু আলাইকুম। আপনাকে স্বাগতম। অনুগ্রহ করে আপনার phone number টি বলুন।",
            "category": "Mixed",
            "lang_code": "ben_Beng",
        },
        {
            "id": "usr_02",
            "text": "আপনার phone number হলো 01672575481",
            "category": "Numeric",
            "lang_code": "ben_Beng",
        },
        {
            "id": "usr_03",
            "text": "আপনার email address টি হলো abushuvom@gmail.com",
            "category": "Mixed",
            "lang_code": "ben_Beng",
        },
        {
            "id": "usr_04",
            "text": "আপনার registration সম্পন্ন হয়েছে , ধন্যবাদ",
            "category": "Mixed",
            "lang_code": "ben_Beng",
        },
        # Numeric / Dates
        {
            "id": "num_01",
            "text": "আমার ফোন নম্বর হল ০১৭-১২৩৪৫৬৭৮।",
            "category": "Numeric",
            "lang_code": "ben_Beng",
        },
        {
            "id": "num_02",
            "text": "আজকের তারিখ ১২ই ডিসেম্বর, ২০২৫।",
            "category": "Numeric",
            "lang_code": "ben_Beng",
        },
        {
            "id": "num_03",
            "text": "দাম মাত্র ৫০০ টাকা।",
            "category": "Numeric",
            "lang_code": "ben_Beng",
        },
        {
            "id": "num_04",
            "text": "The total cost is 4500 BDT.",
            "category": "Numeric",
            "lang_code": "ben_Beng",
        },
        {
            "id": "num_05",
            "text": "১৯৭১ সালে বাংলাদেশ স্বাধীন হয়।",
            "category": "Numeric",
            "lang_code": "ben_Beng",
        },
    ]

    @staticmethod
    def get_prompts():
        """Return the list of prompts."""
        return DatasetManager.PROMPTS

    @staticmethod
    def save_recording(audio_file, prompt_id: str, transcript: str, lang_code: str):
        """Save the recorded audio and update metadata."""
        if not audio_file:
            raise ValueError("No audio file provided")

        prompt = next((p for p in DatasetManager.PROMPTS if p["id"] == prompt_id), None)
        if not prompt:
            # Allow custom prompts if we ever need them, but warn or default
            category = "Custom"
        else:
            category = prompt["category"]

        # Generate unique filename
        filename = f"{prompt_id}_{uuid.uuid4().hex[:8]}.wav"
        save_path = RAW_AUDIO_DIR / filename

        # Save Audio (assuming it comes as blob/wav from frontend)
        # Note: Frontend sending webm/wav blob. ffmpeg might be needed if format issues arise.
        # For simplicity, we save what we get, but best to normalize to wav 16k mono later or now.
        # Let's try to save directly first.
        audio_file.save(str(save_path))

        # Convert to proper WAV 16kHz Mono for training consistency immediately
        try:
            # Temp renaming for conversion source
            temp_src = save_path.with_suffix(".tmp")
            save_path.rename(temp_src)
            convert_audio_to_wav(temp_src, save_path)
            temp_src.unlink()
        except Exception as e:
            print(f"Warning: Audio conversion failed, keeping original: {e}")
            # If conversion fails, we might still have the renamed tmp file?
            # convert_audio_to_wav handles clean up?
            # Actually convert_audio_to_wav takes input and output.
            # If it failed, we might have lost the file if we aren't careful.
            # Let's rely on standard save first.
            if temp_src.exists():
                temp_src.rename(save_path)  # Restore

        # Append to metadata.jsonl
        metadata_entry = {
            "file_name": str(filename),
            "text": transcript,
            "lang_code": lang_code,
            "category": category,
            "prompt_id": prompt_id,
            "timestamp": datetime.now().isoformat(),
        }

        with open(METADATA_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(metadata_entry, ensure_ascii=False) + "\n")

        return metadata_entry


# Supported audio formats by libsndfile (used by fairseq2)
SUPPORTED_FORMATS = {".wav", ".flac", ".ogg", ".au", ".aiff", ".mp3"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024 * 1024  # 1GB max file size
app.config["UPLOAD_FOLDER"] = Path(tempfile.gettempdir()) / "omnilingual_asr_uploads"
app.config["HISTORY_FILE"] = Path.home() / ".omnilingual_asr_history.json"

# Ensure upload directory exists
app.config["UPLOAD_FOLDER"].mkdir(parents=True, exist_ok=True)

# Initialize pipeline at startup
_pipeline: Optional[ASRInferencePipeline] = None
DEFAULT_MODEL_CARD = "omniASR_LLM_Unlimited_300M_v2_local"

# Initialize diarization service
_diarization_service: Optional[DiarizationService] = None
DEFAULT_DIARIZATION_CONFIG = DiarizationConfig(
    model_name="pyannote/speaker-diarization-3.1",
    device="auto",
    min_speakers=1,
    max_speakers=8,
    segment_merge_threshold=0.5,
)


def load_available_models() -> dict:
    """
    Dynamically load available models from the omniasr_local.yaml model card file.
    Returns a dict of { "Display Name": "model_card_name" }.
    """
    import yaml

    yaml_path = (
        Path(__file__).parent
        / "src"
        / "omnilingual_asr"
        / "cards"
        / "models"
        / "omniasr_local.yaml"
    )
    models = {}

    try:
        with open(yaml_path, "r", encoding="utf-8") as f:
            # Parse multi-document YAML (separated by ---)
            for doc in yaml.safe_load_all(f):
                # Skip tokenizers (they have tokenizer_family instead of model_family)
                if (
                    doc
                    and "name" in doc
                    and "model_family" in doc
                    and "tokenizer_family" not in doc
                ):
                    model_name = doc["name"]
                    # Create human-readable display name from model card name
                    # e.g., "omniASR_LLM_Unlimited_7B_v2_local" -> "LLM Unlimited 7B v2"
                    display_name = (
                        model_name.replace("omniASR_", "")
                        .replace("_local", "")
                        .replace("_", " ")
                    )
                    models[display_name] = model_name
    except Exception as e:
        print(f"Warning: Could not load models from YAML: {e}")
        # Fallback to a minimal default
        models = {"LLM Unlimited 300M v2": "omniASR_LLM_Unlimited_300M_v2_local"}

    return models


# Dynamically loaded from omniasr_local.yaml
AVAILABLE_MODELS = load_available_models()

# Common language codes for the dropdown
COMMON_LANGUAGES = [
    ("auto", "Auto Detect (Identify Language)"),
    ("ben_Beng", "Bengali (বাংলা)"),
    ("eng_Latn", "English"),
    ("hin_Deva", "Hindi (हिन्दी)"),
    ("spa_Latn", "Spanish (Español)"),
    ("fra_Latn", "French (Français)"),
    ("deu_Latn", "German (Deutsch)"),
    ("jpn_Jpan", "Japanese (日本語)"),
    ("kor_Hang", "Korean (한국어)"),
    ("zho_Hans", "Chinese Simplified (简体中文)"),
    ("ara_Arab", "Arabic (العربية)"),
    ("por_Latn", "Portuguese (Português)"),
    ("rus_Cyrl", "Russian (Русский)"),
    ("ita_Latn", "Italian (Italiano)"),
    ("nld_Latn", "Dutch (Nederlands)"),
    ("pol_Latn", "Polish (Polski)"),
    ("tur_Latn", "Turkish (Türkçe)"),
    ("vie_Latn", "Vietnamese (Tiếng Việt)"),
    ("tha_Thai", "Thai (ไทย)"),
    ("ind_Latn", "Indonesian (Bahasa Indonesia)"),
    ("msa_Latn", "Malay (Bahasa Melayu)"),
]


def get_pipeline() -> ASRInferencePipeline:
    """Get the pipeline (should be initialized at startup)."""
    global _pipeline
    if _pipeline is None:
        raise RuntimeError("Pipeline not initialized. This should not happen.")
    return _pipeline


def initialize_pipeline(model_card: str = DEFAULT_MODEL_CARD) -> None:
    """Initialize the pipeline at startup or switch models."""
    global _pipeline, DEFAULT_MODEL_CARD

    # If the requested model is already loaded, do nothing
    if _pipeline is not None and DEFAULT_MODEL_CARD == model_card:
        print(f"Pipeline with model '{model_card}' is already loaded.")
        return

    print(f"Loading pipeline with model card '{model_card}'...")
    print("This may take a few moments...")
    try:
        # Force garbage collection if replacing an existing pipeline
        if _pipeline is not None:
            del _pipeline
            import gc

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        _pipeline = ASRInferencePipeline(model_card=model_card)
        DEFAULT_MODEL_CARD = model_card  # Update current model card
        print("✓ Pipeline loaded successfully!")
    except Exception as e:
        print(f"✗ Failed to load pipeline: {e}")
        # If we failed to load the new one, we might be in a bad state (no pipeline).
        # Depending on requirements, we could try to reload the old one or just raise.
        # For now, we raise, leaving _pipeline as None (or deleted).
        _pipeline = None
        raise


def load_history() -> List[dict]:
    """Load transcription history from file."""
    if app.config["HISTORY_FILE"].exists():
        try:
            with open(app.config["HISTORY_FILE"], "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def save_history(history: List[dict]) -> None:
    """Save transcription history to file."""
    # Keep only last 100 entries
    history = history[-100:]
    try:
        with open(app.config["HISTORY_FILE"], "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
    except Exception:
        pass  # Silently fail if we can't save history


def add_to_history(filename: str, lang_code: str, transcription: str) -> None:
    """Add a transcription to history."""
    history = load_history()
    history.append(
        {
            "id": len(history),
            "filename": filename,
            "lang_code": lang_code,
            "transcription": transcription,
            "timestamp": datetime.now().isoformat(),
        }
    )
    save_history(history)


def convert_audio_to_wav(input_file: Path, output_file: Optional[Path] = None) -> Path:
    """
    Convert an audio file to WAV format using ffmpeg.

    Args:
        input_file: Path to the input audio file
        output_file: Optional path for output file. If None, creates a temp file.

    Returns:
        Path to the converted WAV file
    """
    if output_file is None:
        # Create a temporary WAV file
        temp_fd, temp_path = tempfile.mkstemp(suffix=".wav")
        output_file = Path(temp_path)
        # Close the file descriptor so ffmpeg can write to it
        os.close(temp_fd)

    try:
        subprocess.run(
            [
                "/usr/bin/ffmpeg",
                "-i",
                str(input_file),
                "-ar",
                "16000",  # Resample to 16kHz (model requirement)
                "-ac",
                "1",  # Convert to mono
                "-y",  # Overwrite output file
                str(output_file),
            ],
            check=True,
            capture_output=True,
        )
        return output_file
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Failed to convert {input_file} to WAV format. "
            f"ffmpeg error: {e.stderr.decode() if e.stderr else 'Unknown error'}"
        ) from e
    except FileNotFoundError:
        raise RuntimeError(
            "ffmpeg not found. Please install ffmpeg to convert audio files, "
            "or convert your audio files to WAV format manually."
        )


def split_audio_file(input_file: Path, chunk_duration: int = 40) -> List[Path]:
    """
    Split audio file into chunks of specified duration.

    Args:
        input_file: Path to input audio file
        chunk_duration: Duration of each chunk in seconds (default: 40)

    Returns:
        List of paths to generated chunk files
    """
    output_pattern = input_file.parent / f"{input_file.stem}_chunk_%03d.wav"

    try:
        subprocess.run(
            [
                "ffmpeg",
                "-i",
                str(input_file),
                "-f",
                "segment",
                "-segment_time",
                str(chunk_duration),
                "-c",
                "copy",
                str(output_pattern),
            ],
            check=True,
            capture_output=True,
        )

        # Collect generated chunks
        chunks = sorted(list(input_file.parent.glob(f"{input_file.stem}_chunk_*.wav")))
        return chunks
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Failed to split audio: {e.stderr.decode() if e.stderr else str(e)}"
        )


@app.route("/api/transcribe_long", methods=["POST"])
def transcribe_long():
    """Handle long audio file upload and chunked transcription."""
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    lang_code = request.form.get("lang_code", "ben_Beng")

    # Handle Auto Detect
    if lang_code == "auto":
        lang_code = None

    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    # Save uploaded file temporarily
    filename = secure_filename(file.filename)
    original_ext = Path(filename).suffix.lower()
    temp_dir = Path(tempfile.mkdtemp())
    temp_path = (
        temp_dir
        / f"{Path(filename).stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{original_ext}"
    )

    try:
        file.save(str(temp_path))

        # Convert to WAV if needed (for consistent splitting)
        process_path = temp_path
        if original_ext not in SUPPORTED_FORMATS or original_ext != ".wav":
            # Always convert to 16k mono wav for consistency before splitting
            wav_path = temp_path.with_suffix(".wav")
            convert_audio_to_wav(temp_path, wav_path)
            process_path = wav_path

        # Split into chunks
        chunks = split_audio_file(process_path, chunk_duration=40)

        if not chunks:
            return jsonify({"error": "Failed to create audio chunks"}), 500

        pipeline = get_pipeline()
        results = []

        # Transcribe each chunk
        for i, chunk in enumerate(chunks):
            try:
                chunk_path = str(chunk.absolute())
                print(f"Processing chunk {i + 1}/{len(chunks)}: {chunk_path}")
                if not chunk.exists():
                    print(f"Error: Chunk file not found: {chunk_path}")
                    continue

                transcriptions = pipeline.transcribe(
                    [chunk_path], lang=[lang_code], batch_size=1
                )
                if transcriptions:
                    print(f"  Result: {transcriptions[0][:30]}...")
                    results.append(
                        {
                            "segment": i + 1,
                            "filename": chunk.name,
                            "text": transcriptions[0],
                        }
                    )
            except Exception as e:
                print(f"Error transcribing chunk {chunk.name}: {e}")
                results.append(
                    {
                        "segment": i + 1,
                        "filename": chunk.name,
                        "text": f"[Error: {str(e)}]",
                    }
                )

        return jsonify(
            {
                "success": True,
                "results": results,
                "filename": filename,
                "lang_code": lang_code,
            }
        )

    except Exception as e:
        import traceback

        traceback.print_exc()
        return jsonify({"error": f"Long audio processing failed: {str(e)}"}), 500

    finally:
        # Cleanup temp directory
        try:
            shutil.rmtree(str(temp_dir))
        except Exception as e:
            print(f"Cleanup warning: {e}")


def get_diarization_service() -> DiarizationService:
    """Get diarization service (lazy initialization)."""
    global _diarization_service
    if _diarization_service is None:
        print("Initializing diarization service...")
        _diarization_service = DiarizationService(DEFAULT_DIARIZATION_CONFIG)
        print("✓ Diarization service initialized!")
    return _diarization_service


def add_diarization_to_history(
    filename: str, lang_code: str, diarization_result: dict
) -> None:
    """Add a diarized transcription to history."""
    history = load_history()
    history.append(
        {
            "id": len(history),
            "filename": filename,
            "lang_code": lang_code,
            "transcription": diarization_result.get("full_transcript", ""),
            "diarization": diarization_result,
            "timestamp": datetime.now().isoformat(),
        }
    )
    save_history(history)


def format_sse_event(event_type: str, data: dict) -> str:
    """Format data as a Server-Sent Event.

    Args:
        event_type: Type of event (progress, segment, complete, error)
        data: Event data to serialize as JSON

    Returns:
        SSE-formatted string
    """
    # Inject event type into data payload for frontend parser compatibility
    data = data.copy()
    data["event"] = event_type
    json_data = json.dumps(data, ensure_ascii=False)
    return f"event: {event_type}\ndata: {json_data}\n\n"


@app.route("/api/transcribe_with_diarization", methods=["POST"])
def transcribe_with_diarization():
    """Handle audio file upload and transcription with speaker diarization.

    Returns an SSE stream with the following event types:
    - progress: Progress updates during diarization and transcription
    - segment: Completed transcription segments as they're processed
    - complete: Final result with all segments and full transcript
    - error: Error information if processing fails
    """
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    lang_code = request.form.get("lang_code", "ben_Beng")
    num_speakers = request.form.get("num_speakers", type=int)

    # Handle Auto Detect
    if lang_code == "auto":
        lang_code = None

    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    # Get diarization service
    diarization_service = get_diarization_service()

    # Check if diarization is available
    if not diarization_service.is_available():
        return jsonify(
            {"error": "Diarization not available. HuggingFace token not configured."}
        ), 400

    # Save uploaded file temporarily
    filename = secure_filename(file.filename)
    # Preserve original extension
    original_ext = Path(filename).suffix.lower()
    temp_path = (
        app.config["UPLOAD_FOLDER"]
        / f"{Path(filename).stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{original_ext}"
    )
    file.save(str(temp_path))

    # Convert unsupported formats to WAV
    converted_path = None
    audio_path = temp_path

    if original_ext not in SUPPORTED_FORMATS:
        try:
            print(f"Converting {original_ext} file to WAV format...")
            converted_path = convert_audio_to_wav(temp_path)
            audio_path = converted_path
        except Exception as e:
            # Clean up original file
            try:
                temp_path.unlink()
            except Exception:
                pass
            return jsonify({"error": f"Audio conversion failed: {str(e)}"}), 400

    def generate_sse_stream():
        """Generator function that yields SSE events for diarization progress.

        Uses SpeakerTurnTranscriptionPipeline for efficient memory management
        and per-turn error handling.
        """
        turn_transcriptions = []
        diarization_result = None

        try:
            # === PHASE 1: Load diarization model ===
            yield format_sse_event("progress", {
                "stage": "loading",
                "progress": 0,
                "message": "Loading diarization model..."
            })

            diarization_service.load_model()

            yield format_sse_event("progress", {
                "stage": "loading",
                "progress": 1.0,
                "message": "Diarization model loaded"
            })

            # === PHASE 2: Run diarization ===
            yield format_sse_event("progress", {
                "stage": "diarizing",
                "progress": 0,
                "message": "Running speaker diarization (this may take several minutes for long files)..."
            })

            print("Running speaker diarization...")
            
            # Setup queues for threaded execution
            progress_queue = queue.Queue()
            result_queue = queue.Queue()
            error_queue = queue.Queue()

            def progress_callback(update):
                progress_queue.put(update)

            def worker():
                try:
                    res = diarization_service.diarize(
                        str(audio_path), 
                        num_speakers=num_speakers,
                        progress_callback=progress_callback
                    )
                    result_queue.put(res)
                except Exception as e:
                    error_queue.put(e)

            # Start worker thread
            t = threading.Thread(target=worker)
            t.start()

            # Consuming progress updates
            while t.is_alive() or not progress_queue.empty():
                try:
                    update = progress_queue.get(timeout=0.1)
                    yield format_sse_event("progress", {
                        "stage": update.stage,
                        "progress": update.progress,
                        "message": update.message
                    })
                except queue.Empty:
                    pass
            
            # Check for errors
            if not error_queue.empty():
                raise error_queue.get()
            
            # Get result
            if result_queue.empty():
                raise RuntimeError("Diarization finished but returned no result")
            
            diarization_result = result_queue.get()

            yield format_sse_event("progress", {
                "stage": "diarizing",
                "progress": 1.0,
                "message": f"Diarization complete: {diarization_result.num_speakers} speakers found",
                "num_speakers": diarization_result.num_speakers,
                "speakers": diarization_result.speakers
            })

            # === PHASE 3: Build speaker turns ===
            chunk_builder = SmartChunkBuilder()
            speaker_turns = chunk_builder.build_speaker_turns(diarization_result)

            # === PHASE 4: Memory management - unload diarization, free GPU ===
            print("Unloading diarization model to free memory for ASR...")
            diarization_service.unload_model()
            free_gpu_memory()  # Explicit GPU memory cleanup

            yield format_sse_event("progress", {
                "stage": "transcribing",
                "progress": 0,
                "message": f"Starting transcription of {len(speaker_turns)} speaker turns...",
                "total_turns": len(speaker_turns),
                "current_turn": 0
            })

            # === PHASE 5: Transcribe using SpeakerTurnTranscriptionPipeline ===
            pipeline = get_pipeline()

            # Create the transcription pipeline for efficient processing
            transcription_pipeline = SpeakerTurnTranscriptionPipeline(
                asr_pipeline=pipeline,
                chunk_builder=chunk_builder,
                batch_size=1,  # Sequential for memory efficiency
                cleanup_temp_files=True,
            )

            print(f"Transcribing {len(speaker_turns)} speaker turns...")

            # Use streaming transcription for SSE events
            for event in transcription_pipeline.transcribe_turns_streaming(
                str(audio_path), speaker_turns, lang_code
            ):
                if isinstance(event, TxProgress):
                    # Progress update from the pipeline
                    if event.stage != "complete":
                        yield format_sse_event("progress", {
                            "stage": "transcribing",
                            "progress": event.progress_value,
                            "message": event.message,
                            "total_turns": event.total_turns,
                            "current_turn": event.current_turn,
                            "speaker": event.speaker
                        })
                        print(f"Progress: {event.message}")
                elif isinstance(event, TranscriptionSegment):
                    # Completed segment
                    segment_data = event.to_dict()
                    turn_transcriptions.append(segment_data)
                    yield format_sse_event("segment", segment_data)

            # === PHASE 6: Compile final result ===
            full_transcript = build_full_transcript(
                [TranscriptionSegment.from_dict(s) for s in turn_transcriptions]
            )

            # Count successful and failed turns
            successful_turns = sum(1 for s in turn_transcriptions if "error" not in s)
            failed_turns = sum(1 for s in turn_transcriptions if "error" in s)

            diarized_result = {
                "segments": turn_transcriptions,
                "full_transcript": full_transcript,
                "num_speakers": diarization_result.num_speakers,
                "total_duration": diarization_result.total_duration,
                "speakers": diarization_result.speakers,
                "successful_turns": successful_turns,
                "failed_turns": failed_turns,
            }

            # Add to history and get history ID
            history_id = len(load_history())
            add_diarization_to_history(filename, lang_code, diarized_result)

            # Emit completion event
            yield format_sse_event("complete", {
                "success": True,
                "diarized_transcription": diarized_result,
                "filename": filename,
                "lang_code": lang_code,
                "history_id": history_id,
            })

        except ValueError as e:
            # Handle errors
            error_msg = str(e)
            if "Hugging Face token" in error_msg:
                yield format_sse_event("error", {
                    "error": "HuggingFace token required for diarization. Please configure in settings."
                })
            else:
                yield format_sse_event("error", {
                    "error": f"Diarization error: {error_msg}"
                })

        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"Diarization error: {error_details}")
            yield format_sse_event("error", {
                "error": f"Diarization failed: {str(e)}"
            })

        finally:
            # Clean up temp files and ensure diarization model is unloaded
            try:
                diarization_service.unload_model()  # Ensure cleanup
                free_gpu_memory()  # Final memory cleanup
                if temp_path.exists():
                    temp_path.unlink()
                if converted_path and converted_path.exists():
                    converted_path.unlink()
            except Exception:
                pass

    return Response(
        generate_sse_stream(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable buffering in nginx
        }
    )


@app.route("/api/settings/diarization", methods=["GET"])
def get_diarization_settings():
    """Get diarization settings status."""
    try:
        diarization_service = get_diarization_service()
        status = diarization_service.get_status()
        return jsonify(status)
    except Exception as e:
        return jsonify({"error": f"Failed to get settings: {str(e)}"}), 500


@app.route("/api/settings/hf_token", methods=["POST"])
def set_hf_token():
    """Set HuggingFace token for diarization."""
    try:
        data = request.get_json()
        if not data or "token" not in data:
            return jsonify({"error": "Token not provided"}), 400

        token = data["token"].strip()
        if not token:
            return jsonify({"error": "Token cannot be empty"}), 400

        # Get diarization service and set token
        diarization_service = get_diarization_service()
        diarization_service.set_hf_token(token)

        return jsonify(
            {"success": True, "message": "HuggingFace token updated successfully"}
        )

    except Exception as e:
        return jsonify({"error": f"Failed to set token: {str(e)}"}), 500


@app.route("/")
def index():
    """Render the main dashboard page."""
    return render_template("index.html", languages=COMMON_LANGUAGES)


@app.route("/api/transcribe", methods=["POST"])
def transcribe():
    """Handle audio file upload and transcription."""
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    lang_code = request.form.get("lang_code", "ben_Beng")

    # Handle Auto Detect
    if lang_code == "auto":
        lang_code = None

    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    # Save uploaded file temporarily
    filename = secure_filename(file.filename)
    # Preserve original extension
    original_ext = Path(filename).suffix.lower()
    temp_path = (
        app.config["UPLOAD_FOLDER"]
        / f"{Path(filename).stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{original_ext}"
    )
    file.save(str(temp_path))

    # Convert unsupported formats to WAV
    converted_path = None
    audio_path = temp_path

    if original_ext not in SUPPORTED_FORMATS:
        try:
            print(f"Converting {original_ext} file to WAV format...")
            converted_path = convert_audio_to_wav(temp_path)
            audio_path = converted_path
        except Exception as e:
            # Clean up original file
            try:
                temp_path.unlink()
            except Exception:
                pass
            return jsonify({"error": f"Audio conversion failed: {str(e)}"}), 400

    try:
        # Get pipeline and transcribe
        # The pipeline handles audio decoding automatically for supported formats
        pipeline = get_pipeline()
        transcriptions = pipeline.transcribe(
            [str(audio_path)], lang=[lang_code], batch_size=1
        )

        if not transcriptions:
            return jsonify({"error": "Transcription failed - no output generated"}), 500

        transcription = transcriptions[0]

        # Add to history and get the history ID
        history = load_history()
        history_id = len(history)
        add_to_history(filename, lang_code, transcription)

        return jsonify(
            {
                "success": True,
                "transcription": transcription,
                "filename": filename,
                "lang_code": lang_code,
                "history_id": history_id,
            }
        )
    except ValueError as e:
        # Handle audio length or format errors
        error_msg = str(e)
        if "Max audio length" in error_msg:
            return jsonify(
                {"error": "Audio file is too long. Maximum length is 40 seconds."}
            ), 400
        return jsonify({"error": f"Audio processing error: {error_msg}"}), 400
    except Exception as e:
        import traceback

        error_details = traceback.format_exc()
        print(f"Transcription error: {error_details}")
        return jsonify({"error": f"Transcription failed: {str(e)}"}), 500
    finally:
        # Clean up temp files
        try:
            if temp_path.exists():
                temp_path.unlink()
            if converted_path and converted_path.exists():
                converted_path.unlink()
        except Exception:
            pass


@app.route("/api/history", methods=["GET"])
def get_history():
    """Get transcription history."""
    history = load_history()
    # Return most recent first
    return jsonify({"history": list(reversed(history))})


@app.route("/api/download/<int:history_id>", methods=["GET"])
def download_transcription(history_id: int):
    """Download a transcription as a text file."""
    history = load_history()
    entry = next((h for h in history if h["id"] == history_id), None)

    if not entry:
        return jsonify({"error": "History entry not found"}), 404

    # Create a temporary text file
    temp_fd, temp_path = tempfile.mkstemp(suffix=".txt", text=True)
    try:
        with os.fdopen(temp_fd, "w", encoding="utf-8") as f:
            f.write(f"Filename: {entry['filename']}\n")
            f.write(f"Language: {entry['lang_code']}\n")
            f.write(f"Timestamp: {entry['timestamp']}\n")
            f.write("\n" + "=" * 50 + "\n\n")
            f.write(entry["transcription"])

        return send_file(
            temp_path,
            as_attachment=True,
            download_name=f"transcription_{entry['filename']}_{entry['id']}.txt",
            mimetype="text/plain",
        )
    except Exception:
        return jsonify({"error": "Failed to create download file"}), 500


@app.route("/api/history/<int:history_id>/export", methods=["GET"])
def export_transcription(history_id: int):
    """Export a diarized transcription in various formats.

    Query parameters:
        format: Export format - 'inline', 'timestamped', 'json', or 'srt' (default: 'inline')

    Returns:
        Formatted transcription content with appropriate Content-Type
    """
    # Get export format and optional speaker labels from query parameters
    export_format = request.args.get("format", "inline").lower()
    speaker_labels_json = request.args.get("speaker_labels")
    
    speaker_mapping = {}
    if speaker_labels_json:
        try:
            speaker_mapping = json.loads(speaker_labels_json)
        except Exception:
            pass # Ignore invalid mapping

    # Validate format
    if export_format not in EXPORT_FORMATS:
        return jsonify({
            "error": f"Unsupported format '{export_format}'. Valid formats: {', '.join(EXPORT_FORMATS)}"
        }), 400

    # Find history entry
    history = load_history()
    entry = next((h for h in history if h["id"] == history_id), None)

    if not entry:
        return jsonify({"error": "History entry not found"}), 404

    # Check if entry has diarization data
    if "diarization" not in entry:
        # Fall back to plain transcription for non-diarized entries
        if export_format == "json":
            return jsonify({
                "metadata": {
                    "filename": entry.get("filename"),
                    "lang_code": entry.get("lang_code"),
                    "timestamp": entry.get("timestamp"),
                },
                "transcription": entry.get("transcription", ""),
            })
        else:
            # Return plain text for other formats
            content = entry.get("transcription", "")
            return Response(
                content,
                mimetype="text/plain; charset=utf-8",
                headers={
                    "Content-Disposition": f"attachment; filename=transcription_{history_id}.txt"
                }
            )

    try:
        # Create DiarizedTranscriptionResult from history entry
        result = DiarizedTranscriptionResult.from_history_entry(entry)
        
        # Apply speaker renames if provided
        if speaker_mapping:
            result.rename_speakers(speaker_mapping)

        # Export in requested format
        content = result.export(export_format)
        content_type = DiarizedTranscriptionResult.get_content_type(export_format)
        file_ext = DiarizedTranscriptionResult.get_file_extension(export_format)

        # Build filename
        base_name = Path(entry.get("filename", "transcription")).stem
        download_filename = f"{base_name}_{history_id}{file_ext}"

        return Response(
            content,
            mimetype=content_type,
            headers={
                "Content-Disposition": f"attachment; filename={download_filename}"
            }
        )

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Export failed: {str(e)}"}), 500


@app.route("/api/languages", methods=["GET"])
def get_languages():
    """Get list of available languages."""
    return jsonify(
        {"languages": [{"code": code, "name": name} for code, name in COMMON_LANGUAGES]}
    )


@app.route("/api/prompts", methods=["GET"])
def get_prompts():
    """Get the list of data collection prompts."""
    return jsonify({"prompts": DatasetManager.get_prompts()})


@app.route("/api/dataset/submit", methods=["POST"])
def submit_dataset_entry():
    """Receive a recorded audio and its transcript."""
    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files["audio"]
    prompt_id = request.form.get("prompt_id")
    transcript = request.form.get("transcript")
    lang_code = request.form.get("lang_code", "ben_Beng")

    if not prompt_id or not transcript:
        return jsonify({"error": "Missing prompt_id or transcript"}), 400

    try:
        entry = DatasetManager.save_recording(
            audio_file, prompt_id, transcript, lang_code
        )
        return jsonify({"success": True, "entry": entry})
    except Exception as e:
        print(f"Dataset submission error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/models", methods=["GET"])
def get_models():
    """Get list of available models and the current active model."""
    return jsonify({"models": AVAILABLE_MODELS, "current_model": DEFAULT_MODEL_CARD})


@app.route("/api/model", methods=["POST"])
def switch_model():
    """Switch the active ASR model."""
    data = request.get_json()
    if not data or "model_card" not in data:
        return jsonify({"error": "Missing model_card"}), 400

    model_card = data["model_card"]

    # Validate model card
    if model_card not in AVAILABLE_MODELS.values():
        return jsonify({"error": "Invalid model card"}), 400

    try:
        initialize_pipeline(model_card)
        return jsonify({"success": True, "current_model": DEFAULT_MODEL_CARD})
    except Exception as e:
        return jsonify({"error": f"Failed to switch model: {str(e)}"}), 500


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Omnilingual ASR web dashboard")
    parser.add_argument(
        "--host",
        default="192.168.88.252",
        help="Host to bind to (default: 192.168.88.252)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="Port to bind to (default: 5000)",
    )
    parser.add_argument(
        "--model-card",
        default=DEFAULT_MODEL_CARD,
        help=f"Model card to use (default: {DEFAULT_MODEL_CARD})",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run in debug mode",
    )

    args = parser.parse_args()

    # Initialize pipeline BEFORE starting Flask server
    print("=" * 60)
    print("Initializing Omnilingual ASR Dashboard")
    print("=" * 60)
    try:
        initialize_pipeline(args.model_card)
    except Exception as e:
        print(f"\n❌ Failed to initialize pipeline: {e}")
        print("Please check your model card configuration and try again.")
        exit(1)

    print(f"\nStarting Flask server on http://{args.host}:{args.port}")
    print("=" * 60)
    print("Dashboard is ready! Open the URL in your browser.")
    print("=" * 60)

    # Disable reloader to avoid threading issues with fairseq2
    # The model is loaded in the main thread, and we want to keep it there
    # In production, use a proper WSGI server like gunicorn with a single worker
    app.run(
        host=args.host,
        port=args.port,
        debug=args.debug,
        use_reloader=False,  # Disable reloader to avoid threading issues
        threaded=False,  # Disable threading to avoid fairseq2 gang issues
    )

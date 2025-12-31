"""
Logging and error handling utilities for FL Song Gen.
Provides structured logging and custom exceptions with helpful error messages.
"""

import logging
import sys
from typing import Optional


# Configure logger for FL Song Gen
def get_logger(name: str = "FL_SongGen") -> logging.Logger:
    """
    Get a configured logger for FL Song Gen.

    Args:
        name: Logger name (default: FL_SongGen)

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)

    # Only configure if not already configured
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(
            "[%(name)s] %(levelname)s: %(message)s"
        ))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    return logger


# Default logger instance
logger = get_logger()


# Custom Exceptions with helpful messages
class SongGenError(Exception):
    """Base exception for FL Song Gen errors."""
    pass


class ModelNotFoundError(SongGenError):
    """Raised when model files are missing."""

    def __init__(self, variant: str, missing_files: list, model_path: str):
        self.variant = variant
        self.missing_files = missing_files
        self.model_path = model_path

        message = (
            f"Model files missing for '{variant}'.\n"
            f"Missing: {', '.join(missing_files)}\n"
            f"Expected at: {model_path}\n\n"
            f"The model should download automatically. If it fails:\n"
            f"1. Check your internet connection\n"
            f"2. Ensure you have write access to the models directory\n"
            f"3. Try manually downloading from HuggingFace"
        )
        super().__init__(message)


class CheckpointCorruptedError(SongGenError):
    """Raised when checkpoint files are corrupted."""

    def __init__(self, file_path: str, expected_hash: Optional[str] = None):
        self.file_path = file_path
        self.expected_hash = expected_hash

        message = (
            f"Checkpoint file appears to be corrupted: {file_path}\n\n"
            f"Try these steps:\n"
            f"1. Delete the corrupted file\n"
            f"2. Restart ComfyUI to re-download\n"
            f"3. If the problem persists, check your disk space"
        )
        super().__init__(message)


class InsufficientVRAMError(SongGenError):
    """Raised when there's not enough VRAM."""

    def __init__(self, required_gb: float, available_gb: float, variant: str):
        self.required_gb = required_gb
        self.available_gb = available_gb
        self.variant = variant

        message = (
            f"Insufficient VRAM for model '{variant}'.\n"
            f"Required: ~{required_gb:.1f} GB\n"
            f"Available: {available_gb:.1f} GB\n\n"
            f"Try these solutions:\n"
            f"1. Enable 'low_mem' mode in the Model Loader node\n"
            f"2. Use a smaller model variant\n"
            f"3. Close other GPU applications\n"
            f"4. Reduce generation duration"
        )
        super().__init__(message)


class LyricsFormatError(SongGenError):
    """Raised when lyrics format is invalid."""

    def __init__(self, issue: str, example: str = ""):
        self.issue = issue
        self.example = example

        message = (
            f"Invalid lyrics format: {issue}\n\n"
            f"Expected format:\n"
            f"  [intro-short] ; [verse] Line one. Line two ; [chorus] Chorus line ; [outro-short]\n\n"
            f"Rules:\n"
            f"  - Sections separated by ' ; ' (space-semicolon-space)\n"
            f"  - Lines within sections separated by '.'\n"
            f"  - Use section tags: [verse], [chorus], [bridge], [intro-*], [outro-*]"
        )
        if example:
            message += f"\n\nExample:\n{example}"
        super().__init__(message)


class AudioProcessingError(SongGenError):
    """Raised when audio processing fails."""

    def __init__(self, operation: str, details: str = ""):
        self.operation = operation
        self.details = details

        message = f"Audio processing failed during '{operation}'"
        if details:
            message += f": {details}"
        super().__init__(message)


class DemucsNotAvailableError(SongGenError):
    """Raised when Demucs model is not available for style transfer."""

    def __init__(self, model_path: str):
        self.model_path = model_path

        message = (
            f"Demucs separator model not found.\n"
            f"Expected at: {model_path}\n\n"
            f"Style transfer requires the Demucs model for audio separation.\n"
            f"The model should download automatically on first use.\n\n"
            f"If automatic download fails:\n"
            f"1. Check your internet connection\n"
            f"2. Download manually from HuggingFace:\n"
            f"   https://huggingface.co/tencent/SongGeneration/blob/main/third_party/demucs/ckpt/htdemucs.pth"
        )
        super().__init__(message)

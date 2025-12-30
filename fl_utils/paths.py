"""
Path management for FL Song Gen.
Handles model directories and file locations.
"""

import os
from pathlib import Path


def get_comfyui_root() -> Path:
    """Get the ComfyUI root directory."""
    # Navigate from fl_utils -> ComfyUI_FL-SongGen -> custom_nodes -> ComfyUI
    current_dir = Path(__file__).parent
    comfyui_root = current_dir.parent.parent.parent
    return comfyui_root


def get_songgen_models_dir() -> Path:
    """
    Get the directory for SongGeneration models.
    Returns: Path to ComfyUI/models/songgen/
    """
    models_dir = get_comfyui_root() / "models" / "songgen"
    models_dir.mkdir(parents=True, exist_ok=True)
    return models_dir


def get_model_variant_dir(variant: str) -> Path:
    """
    Get directory for a specific model variant.

    Args:
        variant: Model variant name (e.g., 'songgeneration_base')

    Returns:
        Path to the model variant directory
    """
    path = get_songgen_models_dir() / variant
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_auto_prompts_path() -> Path:
    """Get path to auto-style prompt tokens file."""
    return get_songgen_models_dir() / "new_prompt.pt"


def get_demucs_dir() -> Path:
    """Get directory for Demucs separator models."""
    path = get_songgen_models_dir() / "demucs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_songgen_repo_path() -> Path:
    """
    Get path to the SongGeneration repository.
    Expected at: ComfyUI/../SongGeneration/
    """
    comfyui_root = get_comfyui_root()
    # Go up one level from ComfyUI to the dev environment
    dev_env = comfyui_root.parent
    songgen_path = dev_env / "SongGeneration"

    if not songgen_path.exists():
        raise FileNotFoundError(
            f"SongGeneration repository not found at {songgen_path}. "
            "Please clone the repository to this location."
        )

    return songgen_path


def check_model_files(variant: str) -> dict:
    """
    Check if required model files exist for a variant.

    Returns:
        dict with 'exists' bool and 'missing' list of missing files
    """
    variant_dir = get_model_variant_dir(variant)
    required_files = ['config.yaml', 'model.pt']

    missing = []
    for f in required_files:
        if not (variant_dir / f).exists():
            missing.append(f)

    return {
        'exists': len(missing) == 0,
        'missing': missing,
        'path': variant_dir
    }

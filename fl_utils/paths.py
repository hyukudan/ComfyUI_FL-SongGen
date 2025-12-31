"""
Path management for FL Song Gen.
Handles model directories and file locations.
Uses bundled code (codeclm, third_party) - no external repos needed.
Only model checkpoints need to be downloaded.
"""

import os
import sys
from pathlib import Path


def get_comfyui_root() -> Path:
    """Get the ComfyUI root directory."""
    # Navigate from fl_utils -> ComfyUI_FL-SongGen -> custom_nodes -> ComfyUI
    current_dir = Path(__file__).parent
    comfyui_root = current_dir.parent.parent.parent
    return comfyui_root


def get_package_root() -> Path:
    """Get the FL-SongGen package root directory."""
    return Path(__file__).parent.parent


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


def get_checkpoints_dir() -> Path:
    """
    Get path to the checkpoints directory.
    This is where tokenizer/VAE models are stored.
    Downloaded from HuggingFace to: models/songgen/ckpt/
    """
    return get_songgen_models_dir() / "ckpt"


def get_auto_prompts_path() -> Path:
    """Get path to auto-style prompt tokens file."""
    return get_checkpoints_dir() / "new_prompt.pt"


def get_demucs_dir() -> Path:
    """Get directory for Demucs separator models."""
    # Demucs config is bundled, model weights are in checkpoints
    return get_checkpoints_dir() / "demucs"


def get_bundled_codeclm_path() -> Path:
    """Get path to bundled codeclm package."""
    return get_package_root() / "codeclm"


def get_bundled_third_party_path() -> Path:
    """Get path to bundled third_party dependencies."""
    return get_package_root() / "third_party"


def setup_bundled_imports():
    """
    Set up Python path to import from the bundled packages.
    This adds the bundled codeclm and third_party to sys.path.
    """
    package_root = get_package_root()

    # Add bundled package paths
    paths_to_add = [
        str(package_root),  # For 'codeclm' and 'third_party' imports
        str(package_root / "codeclm"),
        str(package_root / "codeclm" / "tokenizer" / "Flow1dVAE"),
        str(package_root / "third_party"),
    ]

    for path in paths_to_add:
        if path not in sys.path and Path(path).exists():
            sys.path.insert(0, path)


def check_bundled_files() -> dict:
    """
    Check if bundled code files are present.

    Returns:
        dict with 'exists' bool and 'missing' list
    """
    package_root = get_package_root()

    required_dirs = [
        package_root / "codeclm",
        package_root / "codeclm" / "models",
        package_root / "codeclm" / "tokenizer",
        package_root / "third_party",
        package_root / "third_party" / "stable_audio_tools",
        package_root / "third_party" / "demucs",
    ]

    missing = []
    for d in required_dirs:
        if not d.exists():
            missing.append(str(d.relative_to(package_root)))

    return {
        'exists': len(missing) == 0,
        'missing': missing,
        'path': package_root
    }


def check_checkpoint_files() -> dict:
    """
    Check if required checkpoint files exist.
    These are the model weights that need to be downloaded.

    Returns:
        dict with 'exists' bool and 'missing' list
    """
    ckpt_dir = get_checkpoints_dir()

    # Core checkpoint directories needed for inference
    required_items = [
        ckpt_dir / "model_1rvq",
        ckpt_dir / "model_septoken",
        ckpt_dir / "vae",
    ]

    missing = []
    for item in required_items:
        if not item.exists():
            missing.append(item.name)

    return {
        'exists': len(missing) == 0,
        'missing': missing,
        'path': ckpt_dir
    }


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


# Minimum expected file sizes in bytes (to detect corrupted downloads)
# These are conservative minimums - actual files are much larger
MIN_FILE_SIZES = {
    'model.pt': 100 * 1024 * 1024,  # 100 MB minimum (actual ~2-5 GB)
    'config.yaml': 1024,  # 1 KB minimum
    'htdemucs.pth': 50 * 1024 * 1024,  # 50 MB minimum
}


def verify_file_integrity(file_path: Path, file_type: str = None) -> dict:
    """
    Basic integrity check for downloaded files.
    Verifies file exists and has reasonable size.

    Args:
        file_path: Path to file to verify
        file_type: Optional type hint for size validation

    Returns:
        dict with 'valid' bool, 'size' in bytes, and 'issue' if any
    """
    if not file_path.exists():
        return {'valid': False, 'size': 0, 'issue': 'File does not exist'}

    file_size = file_path.stat().st_size

    if file_size == 0:
        return {'valid': False, 'size': 0, 'issue': 'File is empty (0 bytes)'}

    # Check minimum size if we know the file type
    if file_type is None:
        file_type = file_path.name

    min_size = MIN_FILE_SIZES.get(file_type, 0)
    if min_size > 0 and file_size < min_size:
        return {
            'valid': False,
            'size': file_size,
            'issue': f'File too small ({file_size / 1024 / 1024:.1f} MB < {min_size / 1024 / 1024:.1f} MB expected)'
        }

    return {'valid': True, 'size': file_size, 'issue': None}


def check_model_integrity(variant: str) -> dict:
    """
    Check integrity of model files for a variant.

    Returns:
        dict with 'valid' bool and list of 'issues' if any
    """
    variant_dir = get_model_variant_dir(variant)
    issues = []

    for filename in ['config.yaml', 'model.pt']:
        file_path = variant_dir / filename
        result = verify_file_integrity(file_path, filename)
        if not result['valid']:
            issues.append(f"{filename}: {result['issue']}")

    return {
        'valid': len(issues) == 0,
        'issues': issues,
        'path': variant_dir
    }


# Legacy aliases for compatibility
def get_runtime_dir() -> Path:
    """Legacy: Get checkpoints directory."""
    return get_checkpoints_dir()


def setup_runtime_imports():
    """Legacy: Set up bundled imports."""
    setup_bundled_imports()


def check_runtime_files() -> dict:
    """Legacy: Check bundled files."""
    return check_bundled_files()


def get_songgen_repo_path() -> Path:
    """Legacy: Get package root."""
    return get_package_root()

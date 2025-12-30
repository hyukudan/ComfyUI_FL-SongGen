"""
Model management for FL Song Gen.
Handles model loading, caching, and configuration.
"""

import os
import sys
import gc
import importlib.util
from pathlib import Path
from typing import Dict, Any, Optional

import torch
from omegaconf import OmegaConf

# Get the fl_utils directory (same directory as this file)
_FL_UTILS_DIR = os.path.dirname(__file__)

# Import paths module explicitly from our package to avoid conflicts
def _import_paths():
    """Import paths module from our fl_utils directory specifically."""
    module_path = os.path.join(_FL_UTILS_DIR, "paths.py")
    spec = importlib.util.spec_from_file_location("songgen_paths", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

_paths = _import_paths()
get_model_variant_dir = _paths.get_model_variant_dir
get_songgen_repo_path = _paths.get_songgen_repo_path
get_auto_prompts_path = _paths.get_auto_prompts_path
get_demucs_dir = _paths.get_demucs_dir
check_model_files = _paths.check_model_files

# Model variant configurations with HuggingFace repo info
MODEL_VARIANTS = {
    "songgeneration_base": {
        "max_duration": 150,  # 2m30s in seconds
        "vram_normal": 16,  # GB
        "vram_low": 10,
        "languages": ["zh"],
        "description": "Base model - Chinese only, 2m30s max",
        "hf_repo": "tencent/SongGeneration",
        "hf_subfolder": "ckpt/songgeneration_base",
    },
    "songgeneration_base_new": {
        "max_duration": 150,
        "vram_normal": 16,
        "vram_low": 10,
        "languages": ["zh", "en"],
        "description": "Base model - Chinese + English, 2m30s max",
        "hf_repo": "lglg666/SongGeneration-base-new",
        "hf_subfolder": None,
    },
    "songgeneration_base_full": {
        "max_duration": 270,  # 4m30s
        "vram_normal": 18,
        "vram_low": 12,
        "languages": ["zh", "en"],
        "description": "Full base model - Chinese + English, 4m30s max",
        "hf_repo": "lglg666/SongGeneration-base-full",
        "hf_subfolder": None,
    },
    "songgeneration_large": {
        "max_duration": 270,
        "vram_normal": 28,
        "vram_low": 22,
        "languages": ["zh", "en"],
        "description": "Large model - Best quality, 4m30s max",
        "hf_repo": "lglg666/SongGeneration-large",
        "hf_subfolder": None,
    }
}

# Runtime files HuggingFace repo
RUNTIME_HF_REPO = "lglg666/SongGeneration-Runtime"

AUTO_STYLE_PRESETS = [
    "Pop", "R&B", "Dance", "Jazz", "Folk",
    "Rock", "Chinese Style", "Chinese Tradition",
    "Metal", "Reggae", "Chinese Opera", "Auto"
]

# Global model cache
_MODEL_CACHE: Dict[str, Any] = {}


def get_variant_list() -> list:
    """Get list of available model variants."""
    return list(MODEL_VARIANTS.keys())


def get_variant_info(variant: str) -> dict:
    """Get information about a model variant."""
    if variant not in MODEL_VARIANTS:
        raise ValueError(f"Unknown variant: {variant}. Available: {list(MODEL_VARIANTS.keys())}")
    return MODEL_VARIANTS[variant].copy()


def clear_model_cache():
    """Clear all cached models and free memory."""
    global _MODEL_CACHE
    _MODEL_CACHE.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _setup_songgen_imports():
    """Add SongGeneration repo to Python path."""
    songgen_path = get_songgen_repo_path()
    songgen_str = str(songgen_path)

    if songgen_str not in sys.path:
        sys.path.insert(0, songgen_str)

    # Also add third_party for demucs
    third_party = songgen_path / "third_party"
    if str(third_party) not in sys.path:
        sys.path.insert(0, str(third_party))


def _register_omegaconf_resolvers():
    """Register OmegaConf resolvers needed for config loading."""
    try:
        OmegaConf.register_new_resolver("eval", lambda x: eval(x), replace=True)
        OmegaConf.register_new_resolver("concat", lambda *x: [xxx for xx in x for xxx in xx], replace=True)
        OmegaConf.register_new_resolver("get_fname", lambda: "songgen", replace=True)
        OmegaConf.register_new_resolver("load_yaml", lambda x: list(OmegaConf.load(x)), replace=True)
    except Exception:
        # Resolvers may already be registered
        pass


def _download_model_files(variant: str) -> bool:
    """
    Download model files from HuggingFace.

    Args:
        variant: Model variant name

    Returns:
        True if download successful
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("[FL SongGen] ERROR: huggingface_hub not installed. Please run: pip install huggingface-hub")
        return False

    if variant not in MODEL_VARIANTS:
        print(f"[FL SongGen] ERROR: Unknown variant: {variant}")
        return False

    variant_info = MODEL_VARIANTS[variant]
    hf_repo = variant_info["hf_repo"]
    hf_subfolder = variant_info.get("hf_subfolder")
    target_dir = get_model_variant_dir(variant)

    print(f"[FL SongGen] Downloading model files for {variant}...")
    print(f"[FL SongGen] From: {hf_repo}")
    print(f"[FL SongGen] To: {target_dir}")

    try:
        # Download only the specific files we need: config.yaml and model.pt
        for filename in ["config.yaml", "model.pt"]:
            target_file = target_dir / filename
            if target_file.exists():
                print(f"[FL SongGen] {filename} already exists, skipping...")
                continue

            if hf_subfolder:
                # File is in a subfolder of the repo
                filepath = f"{hf_subfolder}/{filename}"
            else:
                # File is at root of repo
                filepath = filename

            print(f"[FL SongGen] Downloading {filepath}...")
            hf_hub_download(
                repo_id=hf_repo,
                filename=filepath,
                local_dir=str(target_dir),
                local_dir_use_symlinks=False,
            )

            # If downloaded to subfolder, move to root
            if hf_subfolder:
                import shutil
                src = target_dir / filepath
                if src.exists() and src != target_file:
                    shutil.move(str(src), str(target_file))
                    # Clean up empty subfolder
                    subfolder_path = target_dir / hf_subfolder.split('/')[0]
                    if subfolder_path.exists() and subfolder_path.is_dir():
                        shutil.rmtree(str(subfolder_path))

        print(f"[FL SongGen] Model download complete!")
        return True

    except Exception as e:
        print(f"[FL SongGen] ERROR downloading model: {e}")
        import traceback
        traceback.print_exc()
        return False


# Required runtime files - only download what we actually need
RUNTIME_FILES = {
    # Audio tokenizers
    "ckpt/model_1rvq/model_2_fixed.safetensors": "Audio tokenizer (4.7GB)",
    "ckpt/model_septoken/model_2.safetensors": "Separate tokenizer (4.8GB)",
    # VAE
    "ckpt/vae/autoencoder_music_1320k.ckpt": "VAE model (675MB)",
    "ckpt/vae/stable_audio_1920_vae.json": "VAE config",
    # Encoder
    "ckpt/encode-s12k.pt": "Encoder (4GB)",
    # Content vec model
    "ckpt/models--lengyue233--content-vec-best/blobs/5186a71b15933aca2d9942db95e1aff02642d1f0": "Content vec config",
    "ckpt/models--lengyue233--content-vec-best/blobs/d8dd400e054ddf4e6be75dab5a2549db748cc99e756a097c496c099f65a4854e": "Content vec model (378MB)",
    "ckpt/models--lengyue233--content-vec-best/refs/main": "Content vec ref",
    "ckpt/models--lengyue233--content-vec-best/snapshots/4e9e4560d90e7dbc9035ab4b5582b1da591fd6c3/config.json": "Content vec snapshot config",
    "ckpt/models--lengyue233--content-vec-best/snapshots/4e9e4560d90e7dbc9035ab4b5582b1da591fd6c3/pytorch_model.bin": "Content vec snapshot model",
    # Qwen tokenizer files (small)
    "third_party/Qwen2-7B/config.json": "Qwen config",
    "third_party/Qwen2-7B/generation_config.json": "Qwen generation config",
    "third_party/Qwen2-7B/merges.txt": "Qwen merges",
    "third_party/Qwen2-7B/tokenizer.json": "Qwen tokenizer",
    "third_party/Qwen2-7B/tokenizer_config.json": "Qwen tokenizer config",
    "third_party/Qwen2-7B/vocab.json": "Qwen vocab",
    # Demucs checkpoint
    "third_party/demucs/ckpt/htdemucs.pth": "Demucs model (168MB)",
    "third_party/demucs/ckpt/htdemucs.yaml": "Demucs config",
    # Demucs code files
    "third_party/demucs/__init__.py": "Demucs init",
    "third_party/demucs/run.py": "Demucs run",
    "third_party/demucs/models/__init__.py": "Demucs models init",
    "third_party/demucs/models/apply.py": "Demucs apply",
    "third_party/demucs/models/audio.py": "Demucs audio",
    "third_party/demucs/models/demucs.py": "Demucs model",
    "third_party/demucs/models/htdemucs.py": "Demucs htdemucs",
    "third_party/demucs/models/pretrained.py": "Demucs pretrained",
    "third_party/demucs/models/spec.py": "Demucs spec",
    "third_party/demucs/models/states.py": "Demucs states",
    "third_party/demucs/models/transformer.py": "Demucs transformer",
    "third_party/demucs/models/utils.py": "Demucs utils",
}

# stable_audio_tools files - download the whole folder since there are many
STABLE_AUDIO_FOLDERS = [
    "third_party/stable_audio_tools",
]


def _download_runtime_files() -> bool:
    """
    Download runtime files (ckpt and third_party folders) to SongGeneration repo.
    Only downloads the specific files needed for inference.

    Returns:
        True if download successful
    """
    try:
        from huggingface_hub import hf_hub_download, snapshot_download
    except ImportError:
        print("[FL SongGen] ERROR: huggingface_hub not installed. Please run: pip install huggingface-hub")
        return False

    songgen_path = get_songgen_repo_path()

    # Check if all required files exist
    all_exist = True
    for filepath in RUNTIME_FILES.keys():
        if not (songgen_path / filepath).exists():
            all_exist = False
            break

    if all_exist:
        print("[FL SongGen] Runtime files already exist")
        return True

    print(f"[FL SongGen] Downloading runtime files...")
    print(f"[FL SongGen] From: {RUNTIME_HF_REPO}")
    print(f"[FL SongGen] To: {songgen_path}")

    try:
        # Download individual files
        for filepath, description in RUNTIME_FILES.items():
            target_file = songgen_path / filepath
            if target_file.exists():
                continue

            print(f"[FL SongGen] Downloading {description}...")
            target_file.parent.mkdir(parents=True, exist_ok=True)

            hf_hub_download(
                repo_id=RUNTIME_HF_REPO,
                filename=filepath,
                local_dir=str(songgen_path),
                local_dir_use_symlinks=False,
            )

        # Download stable_audio_tools folder (has many files)
        for folder in STABLE_AUDIO_FOLDERS:
            folder_path = songgen_path / folder
            if folder_path.exists():
                continue

            print(f"[FL SongGen] Downloading {folder}...")
            snapshot_download(
                repo_id=RUNTIME_HF_REPO,
                local_dir=str(songgen_path),
                local_dir_use_symlinks=False,
                allow_patterns=[f"{folder}/**"],
                ignore_patterns=["*.md", ".gitattributes", "*.git*"],
            )

        print(f"[FL SongGen] Runtime files download complete!")
        return True

    except Exception as e:
        print(f"[FL SongGen] ERROR downloading runtime files: {e}")
        import traceback
        traceback.print_exc()
        return False


def ensure_model_files(variant: str) -> bool:
    """
    Ensure model files exist, downloading if necessary.

    Args:
        variant: Model variant name

    Returns:
        True if files exist or were downloaded successfully
    """
    # First check if runtime files exist
    try:
        songgen_path = get_songgen_repo_path()
        ckpt_dir = songgen_path / "ckpt"
        third_party_dir = songgen_path / "third_party"

        if not ckpt_dir.exists() or not third_party_dir.exists():
            print("[FL SongGen] Runtime files missing, downloading...")
            if not _download_runtime_files():
                return False
    except FileNotFoundError:
        print("[FL SongGen] ERROR: SongGeneration repository not found!")
        print("[FL SongGen] Please clone it first:")
        print("  cd /path/to/ComfyUI/..")
        print("  git clone https://github.com/AslpLab/SongGeneration.git")
        return False

    # Check model files
    file_check = check_model_files(variant)
    if file_check['exists']:
        return True

    # Download model files
    print(f"[FL SongGen] Model files missing: {file_check['missing']}")
    return _download_model_files(variant)


def load_model(
    variant: str,
    low_mem: bool = False,
    use_flash_attn: bool = False,
    force_reload: bool = False,
    device: Optional[str] = None
) -> Dict[str, Any]:
    """
    Load SongGeneration model.

    Args:
        variant: Model variant name
        low_mem: Enable low memory mode
        use_flash_attn: Use Flash Attention 2
        force_reload: Force reload even if cached
        device: Device to load model on (default: auto-detect)

    Returns:
        Dict containing model components and configuration
    """
    global _MODEL_CACHE

    cache_key = f"{variant}_{low_mem}_{use_flash_attn}"

    # Return cached model if available
    if not force_reload and cache_key in _MODEL_CACHE:
        print(f"[FL SongGen] Using cached model: {variant}")
        return _MODEL_CACHE[cache_key]

    # Clear cache if force reload
    if force_reload:
        clear_model_cache()

    # Ensure model files exist (download if necessary)
    if not ensure_model_files(variant):
        file_check = check_model_files(variant)
        raise FileNotFoundError(
            f"Model files missing for {variant} at {file_check['path']}. "
            f"Missing: {file_check['missing']}. "
            "Automatic download failed. Please download manually from HuggingFace."
        )

    print(f"[FL SongGen] Loading model: {variant}")
    print(f"[FL SongGen] Low memory mode: {low_mem}")
    print(f"[FL SongGen] Flash Attention: {use_flash_attn}")

    # Setup imports
    _setup_songgen_imports()
    _register_omegaconf_resolvers()

    # Determine device
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
            print("[FL SongGen] WARNING: CUDA not available, using CPU (very slow)")

    # Load configuration
    variant_dir = get_model_variant_dir(variant)
    cfg_path = variant_dir / "config.yaml"
    ckpt_path = variant_dir / "model.pt"

    cfg = OmegaConf.load(str(cfg_path))
    cfg.lm.use_flash_attn_2 = use_flash_attn
    cfg.mode = 'inference'
    max_duration = cfg.max_dur

    # Import SongGeneration modules
    from codeclm.models import builders, CodecLM

    model_info = {
        "variant": variant,
        "config": cfg,
        "max_duration": max_duration,
        "sample_rate": cfg.sample_rate,
        "device": device,
        "low_mem": low_mem,
        "use_flash_attn": use_flash_attn,
    }

    if low_mem:
        # Low memory mode: load components on-demand
        model_info["ckpt_path"] = str(ckpt_path)
        model_info["loaded"] = False
        print("[FL SongGen] Low memory mode: model will be loaded on-demand")
    else:
        # Normal mode: load everything now
        model_info = _load_full_model(model_info, cfg, ckpt_path, device)

    # Load auto prompts if available
    auto_prompts_path = get_auto_prompts_path()
    if auto_prompts_path.exists():
        model_info["auto_prompts"] = torch.load(str(auto_prompts_path), map_location='cpu')
        print(f"[FL SongGen] Loaded auto prompts from {auto_prompts_path}")
    else:
        model_info["auto_prompts"] = None
        print(f"[FL SongGen] Auto prompts not found at {auto_prompts_path}")

    # Cache the model
    _MODEL_CACHE[cache_key] = model_info

    print(f"[FL SongGen] Model loaded successfully")
    return model_info


def _load_full_model(
    model_info: dict,
    cfg: OmegaConf,
    ckpt_path: Path,
    device: str
) -> dict:
    """Load full model (non-low-memory mode)."""
    from codeclm.models import builders, CodecLM

    # Load audio tokenizer for prompt encoding
    print("[FL SongGen] Loading audio tokenizer...")
    audio_tokenizer = builders.get_audio_tokenizer_model(cfg.audio_tokenizer_checkpoint, cfg)
    if audio_tokenizer is not None:
        audio_tokenizer = audio_tokenizer.eval()
        if device == "cuda":
            audio_tokenizer = audio_tokenizer.cuda()
    model_info["audio_tokenizer"] = audio_tokenizer

    # Load separate tokenizer for vocal/bgm encoding
    print("[FL SongGen] Loading separate tokenizer...")
    if "audio_tokenizer_checkpoint_sep" in cfg.keys():
        separate_tokenizer = builders.get_audio_tokenizer_model(cfg.audio_tokenizer_checkpoint_sep, cfg)
        if separate_tokenizer is not None:
            separate_tokenizer = separate_tokenizer.eval()
            if device == "cuda":
                separate_tokenizer = separate_tokenizer.cuda()
    else:
        separate_tokenizer = None
    model_info["separate_tokenizer"] = separate_tokenizer

    # Load LM
    print("[FL SongGen] Loading language model...")
    audiolm = builders.get_lm_model(cfg)
    checkpoint = torch.load(str(ckpt_path), map_location='cpu')
    audiolm_state_dict = {
        k.replace('audiolm.', ''): v
        for k, v in checkpoint.items()
        if k.startswith('audiolm')
    }
    audiolm.load_state_dict(audiolm_state_dict, strict=False)
    audiolm = audiolm.eval()

    if device == "cuda":
        audiolm = audiolm.cuda().to(torch.float16)

    # Create CodecLM wrapper
    model = CodecLM(
        name=model_info["variant"],
        lm=audiolm,
        audiotokenizer=audio_tokenizer,
        max_duration=model_info["max_duration"],
        seperate_tokenizer=separate_tokenizer,
    )

    model_info["model"] = model
    model_info["audiolm"] = audiolm
    model_info["loaded"] = True

    # Cleanup checkpoint to save memory
    del checkpoint
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return model_info


def load_separator(device: str = "cuda") -> Any:
    """
    Load Demucs separator for audio source separation.

    Args:
        device: Device to load on

    Returns:
        Separator instance
    """
    _setup_songgen_imports()

    songgen_path = get_songgen_repo_path()
    dm_model_path = songgen_path / "third_party" / "demucs" / "ckpt" / "htdemucs.pth"
    dm_config_path = songgen_path / "third_party" / "demucs" / "ckpt" / "htdemucs.yaml"

    if not dm_model_path.exists():
        raise FileNotFoundError(
            f"Demucs model not found at {dm_model_path}. "
            "Please ensure the SongGeneration repo includes the Demucs checkpoint."
        )

    # Import and create separator
    from third_party.demucs.models.pretrained import get_model_from_yaml

    demucs_model = get_model_from_yaml(str(dm_config_path), str(dm_model_path))

    if device == "cuda" and torch.cuda.is_available():
        demucs_model = demucs_model.to(torch.device("cuda"))
    else:
        demucs_model = demucs_model.to(torch.device("cpu"))

    demucs_model.eval()

    print("[FL SongGen] Demucs separator loaded")
    return demucs_model


def get_model_status() -> dict:
    """Get status of loaded models."""
    status = {
        "cached_models": list(_MODEL_CACHE.keys()),
        "available_variants": list(MODEL_VARIANTS.keys()),
        "auto_style_presets": AUTO_STYLE_PRESETS,
    }

    # Check VRAM if CUDA available
    if torch.cuda.is_available():
        status["cuda_available"] = True
        status["vram_total_gb"] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        status["vram_allocated_gb"] = torch.cuda.memory_allocated(0) / (1024**3)
        status["vram_reserved_gb"] = torch.cuda.memory_reserved(0) / (1024**3)
    else:
        status["cuda_available"] = False

    return status

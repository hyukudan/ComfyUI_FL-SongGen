"""
FL Song Gen Advanced Generate Node.
Generation with advanced sampling parameters (top_p, etc).
"""

import sys
import os
from typing import Tuple
import importlib.util

from comfy.utils import ProgressBar

# Get the package root directory
_PACKAGE_ROOT = os.path.dirname(os.path.dirname(__file__))

# Import modules explicitly from our package to avoid conflicts
def _import_from_package(module_name, file_name):
    """Import a module from our package specifically."""
    module_path = os.path.join(_PACKAGE_ROOT, "fl_utils", f"{file_name}.py")
    spec = importlib.util.spec_from_file_location(f"songgen_{module_name}", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Import our modules
_songgen_wrapper = _import_from_package("songgen_wrapper", "songgen_wrapper")
_audio_utils = _import_from_package("audio_utils", "audio_utils")

SongGenWrapper = _songgen_wrapper.SongGenWrapper
empty_audio = _audio_utils.empty_audio


class FL_SongGen_AdvancedGenerate:
    """
    Generate songs with advanced sampling parameters.

    Exposes additional controls like top_p (nucleus sampling) for more
    fine-grained control over the generation process.

    Top-P vs Top-K:
    - Top-K: Always samples from the K most likely tokens
    - Top-P: Samples from smallest set of tokens with cumulative probability >= P
    - Set top_p > 0 to use nucleus sampling instead of top-k
    """

    RETURN_TYPES = ("AUDIO", "AUDIO", "AUDIO")
    RETURN_NAMES = ("mixed_audio", "vocal_audio", "bgm_audio")
    FUNCTION = "generate"
    CATEGORY = "FL Song Gen"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (
                    "SONGGEN_MODEL",
                    {
                        "tooltip": "Loaded SongGeneration model"
                    }
                ),
                "lyrics": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "[intro-short] ; [verse] Hello world.This is a test ; [chorus] Singing along.Making music ; [outro-short]",
                        "tooltip": "Formatted lyrics with section tags"
                    }
                ),
            },
            "optional": {
                "description": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Style description (e.g., 'female, pop, emotional, piano')"
                    }
                ),
                "duration": (
                    "FLOAT",
                    {
                        "default": 60.0,
                        "min": 30.0,
                        "max": 270.0,
                        "step": 5.0,
                        "tooltip": "Target duration in seconds"
                    }
                ),
                "temperature": (
                    "FLOAT",
                    {
                        "default": 0.9,
                        "min": 0.1,
                        "max": 2.0,
                        "step": 0.05,
                        "tooltip": "Sampling temperature (higher = more random)"
                    }
                ),
                "cfg_coef": (
                    "FLOAT",
                    {
                        "default": 1.5,
                        "min": 0.5,
                        "max": 5.0,
                        "step": 0.1,
                        "tooltip": "Classifier-free guidance strength"
                    }
                ),
                "top_k": (
                    "INT",
                    {
                        "default": 50,
                        "min": 1,
                        "max": 500,
                        "step": 10,
                        "tooltip": "Top-k sampling (used when top_p=0)"
                    }
                ),
                "top_p": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": "Top-p (nucleus) sampling. Set > 0 to use instead of top_k"
                    }
                ),
                "gen_type": (
                    ["mixed", "separate", "vocal", "bgm"],
                    {
                        "default": "mixed",
                        "tooltip": "Output type: mixed, separate (all tracks), vocal only, or bgm only"
                    }
                ),
                "seed": (
                    "INT",
                    {
                        "default": -1,
                        "min": -1,
                        "max": 2147483647,
                        "tooltip": "Random seed (-1 for random)"
                    }
                ),
            }
        }

    def generate(
        self,
        model: dict,
        lyrics: str,
        description: str = "",
        duration: float = 60.0,
        temperature: float = 0.9,
        cfg_coef: float = 1.5,
        top_k: int = 50,
        top_p: float = 0.0,
        gen_type: str = "mixed",
        seed: int = -1
    ) -> Tuple[dict, dict, dict]:
        """
        Generate song with advanced parameters.
        """
        print(f"\n{'='*60}")
        print(f"[FL SongGen Advanced] Starting Generation")
        print(f"{'='*60}")
        print(f"Duration: {duration}s")
        print(f"Temperature: {temperature}")
        print(f"CFG: {cfg_coef}")
        print(f"Top-K: {top_k}")
        print(f"Top-P: {top_p}")
        sampling_mode = "nucleus (top-p)" if top_p > 0 else "top-k"
        print(f"Sampling Mode: {sampling_mode}")
        print(f"Gen Type: {gen_type}")
        print(f"Seed: {seed}")
        print(f"Description: {description[:50]}..." if description else "Description: None")
        print(f"{'='*60}\n")

        # Validate duration
        max_dur = model.get("max_duration", 150)
        if duration > max_dur:
            print(f"[FL SongGen] WARNING: Duration {duration}s exceeds model max {max_dur}s, clamping.")
            duration = max_dur

        # Create wrapper and set up progress bar
        wrapper = SongGenWrapper(model)

        frame_rate = 25
        total_steps = int(frame_rate * duration)
        pbar = ProgressBar(total_steps)

        def progress_callback(current, total):
            pbar.update_absolute(current)
            if current % 100 == 0:
                print(f"[FL SongGen] Progress: {current}/{total} tokens")

        wrapper.set_progress_callback(progress_callback)

        # Generate
        try:
            mixed_audio, vocal_audio, bgm_audio = wrapper.generate(
                lyrics=lyrics,
                description=description if description.strip() else None,
                duration=duration,
                temperature=temperature,
                cfg_coef=cfg_coef,
                top_k=top_k,
                top_p=top_p,
                gen_type=gen_type,
                seed=seed,
            )

            pbar.update_absolute(total_steps)

            print(f"\n{'='*60}")
            print(f"[FL SongGen Advanced] Generation Complete!")
            print(f"Mixed Audio: {mixed_audio['waveform'].shape}, {mixed_audio['sample_rate']}Hz")
            print(f"{'='*60}\n")

            return (mixed_audio, vocal_audio, bgm_audio)

        except Exception as e:
            print(f"\n{'='*60}")
            print(f"[FL SongGen Advanced] ERROR: Generation failed!")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            print(f"{'='*60}\n")

            sample_rate = model.get("sample_rate", 24000)
            return (
                empty_audio(sample_rate),
                empty_audio(sample_rate),
                empty_audio(sample_rate)
            )

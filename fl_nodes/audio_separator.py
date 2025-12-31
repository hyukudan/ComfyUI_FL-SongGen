"""
FL Song Gen Audio Separator Node.
Separate audio into stems (vocals, drums, bass, other) using Demucs.
"""

import sys
import os
from typing import Tuple
import importlib.util

import torch
import torchaudio

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

_model_manager = _import_from_package("model_manager", "model_manager")
_audio_utils = _import_from_package("audio_utils", "audio_utils")

load_separator = _model_manager.load_separator
empty_audio = _audio_utils.empty_audio


class FL_SongGen_AudioSeparator:
    """
    Separate audio into individual stems using Demucs (htdemucs model).

    Outputs 4 stems: vocals, drums, bass, and other (instruments).
    Useful for extracting vocals for remixing or isolating instrumentals.
    """

    RETURN_TYPES = ("AUDIO", "AUDIO", "AUDIO", "AUDIO")
    RETURN_NAMES = ("vocals", "drums", "bass", "other")
    FUNCTION = "separate"
    CATEGORY = "FL Song Gen"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": (
                    "AUDIO",
                    {
                        "tooltip": "Audio to separate into stems"
                    }
                ),
            },
            "optional": {
                "device": (
                    ["auto", "cuda", "cpu"],
                    {
                        "default": "auto",
                        "tooltip": "Device for processing (auto uses GPU if available)"
                    }
                ),
            }
        }

    def separate(
        self,
        audio: dict,
        device: str = "auto"
    ) -> Tuple[dict, dict, dict, dict]:
        """
        Separate audio into stems.

        Args:
            audio: ComfyUI AUDIO dict with waveform and sample_rate
            device: Processing device

        Returns:
            (vocals, drums, bass, other) as ComfyUI AUDIO dicts
        """
        print(f"\n{'='*60}")
        print(f"[FL SongGen Audio Separator] Starting Separation")
        print(f"{'='*60}")

        waveform = audio["waveform"]
        sample_rate = audio["sample_rate"]

        print(f"Input: {waveform.shape}, {sample_rate}Hz")

        # Determine device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Device: {device}")

        try:
            # Load Demucs separator
            print("[FL SongGen] Loading Demucs separator...")
            separator = load_separator(device=device)

            # Demucs expects 44100Hz
            target_sr = 44100
            if sample_rate != target_sr:
                print(f"[FL SongGen] Resampling {sample_rate}Hz -> {target_sr}Hz")
                waveform = torchaudio.functional.resample(
                    waveform.squeeze(0), sample_rate, target_sr
                ).unsqueeze(0)

            # Ensure stereo (Demucs expects stereo)
            if waveform.dim() == 2:
                waveform = waveform.unsqueeze(0)  # Add batch dim
            if waveform.shape[1] == 1:
                waveform = waveform.repeat(1, 2, 1)  # Mono to stereo

            # Move to device
            waveform = waveform.to(device)

            # Import apply_model from bundled demucs (in third_party)
            # Need to add package root to path for third_party imports
            if _PACKAGE_ROOT not in sys.path:
                sys.path.insert(0, _PACKAGE_ROOT)
            from third_party.demucs.models.apply import apply_model

            # Run separation using apply_model (not direct forward call)
            print("[FL SongGen] Running separation (this may take a while)...")

            # Debug: print model's source names
            if hasattr(separator, 'sources'):
                print(f"[FL SongGen] Model sources: {separator.sources}")

            with torch.no_grad():
                sources = apply_model(
                    separator, waveform,
                    device=device,
                    shifts=1,
                    split=True,
                    overlap=0.25,
                    progress=True
                )

            # Sources shape: [batch, num_sources, channels, samples]
            print(f"[FL SongGen] Output shape: {sources.shape}")
            sources = sources.squeeze(0)  # Remove batch dim

            # Get source names from model to map correctly
            source_names = getattr(separator, 'sources', ['drums', 'bass', 'other', 'vocals'])
            print(f"[FL SongGen] Source order: {source_names}")

            # Create a dict to map by name
            source_dict = {name: sources[i] for i, name in enumerate(source_names)}

            # Extract by name (handle both 'vocal' and 'vocals')
            vocals = source_dict.get('vocals', source_dict.get('vocal', sources[-1]))
            drums = source_dict.get('drums', sources[0])
            bass = source_dict.get('bass', sources[1])
            other = source_dict.get('other', sources[2])

            # Debug: show audio levels (RMS) for each stem
            def rms(x):
                return torch.sqrt(torch.mean(x ** 2)).item()
            print(f"[FL SongGen] Stem levels (RMS):")
            print(f"  - Drums: {rms(drums):.6f}")
            print(f"  - Bass:  {rms(bass):.6f}")
            print(f"  - Other: {rms(other):.6f}")
            print(f"  - Vocal: {rms(vocals):.6f}")

            # Convert back to original sample rate if needed
            if sample_rate != target_sr:
                print(f"[FL SongGen] Resampling back to {sample_rate}Hz")
                vocals = torchaudio.functional.resample(vocals, target_sr, sample_rate)
                drums = torchaudio.functional.resample(drums, target_sr, sample_rate)
                bass = torchaudio.functional.resample(bass, target_sr, sample_rate)
                other = torchaudio.functional.resample(other, target_sr, sample_rate)

            # Debug: show levels after resample
            print(f"[FL SongGen] After resample RMS:")
            print(f"  - Drums: {rms(drums):.6f}")
            print(f"  - Bass:  {rms(bass):.6f}")
            print(f"  - Other: {rms(other):.6f}")
            print(f"  - Vocal: {rms(vocals):.6f}")

            # Move to CPU and format for ComfyUI
            def to_audio_dict(tensor, sr):
                return {
                    "waveform": tensor.unsqueeze(0).cpu(),  # [1, channels, samples]
                    "sample_rate": sr
                }

            vocals_out = to_audio_dict(vocals, sample_rate)
            drums_out = to_audio_dict(drums, sample_rate)
            bass_out = to_audio_dict(bass, sample_rate)
            other_out = to_audio_dict(other, sample_rate)

            # Debug: verify output RMS
            print(f"[FL SongGen] Final output RMS:")
            print(f"  - vocals_out: {rms(vocals_out['waveform']):.6f}")
            print(f"  - drums_out:  {rms(drums_out['waveform']):.6f}")
            print(f"  - bass_out:   {rms(bass_out['waveform']):.6f}")
            print(f"  - other_out:  {rms(other_out['waveform']):.6f}")

            print(f"\n{'='*60}")
            print(f"[FL SongGen Audio Separator] Separation Complete!")
            print(f"Vocals: {vocals_out['waveform'].shape}")
            print(f"Drums: {drums_out['waveform'].shape}")
            print(f"Bass: {bass_out['waveform'].shape}")
            print(f"Other: {other_out['waveform'].shape}")
            print(f"{'='*60}\n")

            return (vocals_out, drums_out, bass_out, other_out)

        except Exception as e:
            print(f"\n{'='*60}")
            print(f"[FL SongGen Audio Separator] ERROR: Separation failed!")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            print(f"{'='*60}\n")

            return (
                empty_audio(sample_rate),
                empty_audio(sample_rate),
                empty_audio(sample_rate),
                empty_audio(sample_rate)
            )

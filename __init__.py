"""
FL Song Gen - AI Song Generation for ComfyUI
Based on Tencent's SongGeneration (LeVo) model

Generate complete songs with vocals and instrumentals from lyrics!
"""

import sys
import os
import warnings
import importlib.util

# Fix for Python 3.12+: pkg_resources.packaging was removed
# CLIP (via k-diffusion) tries to import it, causing ImportError
# This monkey-patch must run BEFORE any module loads CLIP
try:
    import packaging
    import pkg_resources
    # Inject packaging into pkg_resources if not present
    if not hasattr(pkg_resources, 'packaging'):
        pkg_resources.packaging = packaging
except ImportError:
    pass  # packaging not installed, will be handled by requirements

# Suppress cosmetic warnings from transformers about GenerationMixin and checkpointing format
# These need to be set early, before transformers is imported
warnings.filterwarnings("ignore", message=".*GenerationMixin.*")
warnings.filterwarnings("ignore", message=".*old version of the checkpointing format.*")
warnings.filterwarnings("ignore", message=".*doesn't directly inherit from.*")
warnings.filterwarnings("ignore", message=".*will NOT inherit from.*")
warnings.filterwarnings("ignore", message=".*_set_gradient_checkpointing.*")
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.*")

# Get current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

def import_module_from_path(module_name, file_path):
    """Import a module from an explicit file path to avoid naming conflicts."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

# Import utilities first (needed by nodes)
sg_paths = import_module_from_path(
    "sg_paths",
    os.path.join(current_dir, "fl_utils", "paths.py")
)
sg_audio_utils = import_module_from_path(
    "sg_audio_utils",
    os.path.join(current_dir, "fl_utils", "audio_utils.py")
)
sg_model_manager = import_module_from_path(
    "sg_model_manager",
    os.path.join(current_dir, "fl_utils", "model_manager.py")
)
sg_songgen_wrapper = import_module_from_path(
    "sg_songgen_wrapper",
    os.path.join(current_dir, "fl_utils", "songgen_wrapper.py")
)

# Import nodes
sg_model_loader = import_module_from_path(
    "sg_model_loader",
    os.path.join(current_dir, "fl_nodes", "model_loader.py")
)
sg_lyrics_formatter = import_module_from_path(
    "sg_lyrics_formatter",
    os.path.join(current_dir, "fl_nodes", "lyrics_formatter.py")
)
sg_description_builder = import_module_from_path(
    "sg_description_builder",
    os.path.join(current_dir, "fl_nodes", "description_builder.py")
)
sg_generate = import_module_from_path(
    "sg_generate",
    os.path.join(current_dir, "fl_nodes", "generate.py")
)
sg_style_transfer = import_module_from_path(
    "sg_style_transfer",
    os.path.join(current_dir, "fl_nodes", "style_transfer.py")
)
sg_auto_style = import_module_from_path(
    "sg_auto_style",
    os.path.join(current_dir, "fl_nodes", "auto_style.py")
)
sg_audio_separator = import_module_from_path(
    "sg_audio_separator",
    os.path.join(current_dir, "fl_nodes", "audio_separator.py")
)
sg_advanced_generate = import_module_from_path(
    "sg_advanced_generate",
    os.path.join(current_dir, "fl_nodes", "advanced_generate.py")
)

# Get node classes
FL_SongGen_ModelLoader = sg_model_loader.FL_SongGen_ModelLoader
FL_SongGen_LyricsFormatter = sg_lyrics_formatter.FL_SongGen_LyricsFormatter
FL_SongGen_DescriptionBuilder = sg_description_builder.FL_SongGen_DescriptionBuilder
FL_SongGen_Generate = sg_generate.FL_SongGen_Generate
FL_SongGen_StyleTransfer = sg_style_transfer.FL_SongGen_StyleTransfer
FL_SongGen_AutoStyle = sg_auto_style.FL_SongGen_AutoStyle
FL_SongGen_AudioSeparator = sg_audio_separator.FL_SongGen_AudioSeparator
FL_SongGen_AdvancedGenerate = sg_advanced_generate.FL_SongGen_AdvancedGenerate

# Node registration for ComfyUI
NODE_CLASS_MAPPINGS = {
    "FL_SongGen_ModelLoader": FL_SongGen_ModelLoader,
    "FL_SongGen_LyricsFormatter": FL_SongGen_LyricsFormatter,
    "FL_SongGen_DescriptionBuilder": FL_SongGen_DescriptionBuilder,
    "FL_SongGen_Generate": FL_SongGen_Generate,
    "FL_SongGen_StyleTransfer": FL_SongGen_StyleTransfer,
    "FL_SongGen_AutoStyle": FL_SongGen_AutoStyle,
    "FL_SongGen_AudioSeparator": FL_SongGen_AudioSeparator,
    "FL_SongGen_AdvancedGenerate": FL_SongGen_AdvancedGenerate,
}

# Display names for the UI
NODE_DISPLAY_NAME_MAPPINGS = {
    "FL_SongGen_ModelLoader": "FL Song Gen Model Loader",
    "FL_SongGen_LyricsFormatter": "FL Song Gen Lyrics Formatter",
    "FL_SongGen_DescriptionBuilder": "FL Song Gen Description Builder",
    "FL_SongGen_Generate": "FL Song Gen Generate",
    "FL_SongGen_StyleTransfer": "FL Song Gen Style Transfer",
    "FL_SongGen_AutoStyle": "FL Song Gen Auto Style",
    "FL_SongGen_AudioSeparator": "FL Song Gen Audio Separator",
    "FL_SongGen_AdvancedGenerate": "FL Song Gen Advanced Generate",
}

# Version info
__version__ = "1.2.0"

# ASCII banner
ascii_art = """
 ███████╗██╗         ███████╗ ██████╗ ███╗   ██╗ ██████╗      ██████╗ ███████╗███╗   ██╗
 ██╔════╝██║         ██╔════╝██╔═══██╗████╗  ██║██╔════╝     ██╔════╝ ██╔════╝████╗  ██║
 █████╗  ██║         ███████╗██║   ██║██╔██╗ ██║██║  ███╗    ██║  ███╗█████╗  ██╔██╗ ██║
 ██╔══╝  ██║         ╚════██║██║   ██║██║╚██╗██║██║   ██║    ██║   ██║██╔══╝  ██║╚██╗██║
 ██║     ███████╗    ███████║╚██████╔╝██║ ╚████║╚██████╔╝    ╚██████╔╝███████╗██║ ╚████║
 ╚═╝     ╚══════╝    ╚══════╝ ╚═════╝ ╚═╝  ╚═══╝ ╚═════╝      ╚═════╝ ╚══════╝╚═╝  ╚═══╝
"""

print(ascii_art)
print("=" * 85)
print(f"FL Song Gen v{__version__} - AI Song Generation for ComfyUI")
print("Based on Tencent's SongGeneration (LeVo) model")
print("-" * 85)
print("Nodes loaded:")
print("  - FL Song Gen Model Loader       : Load the song generation model")
print("  - FL Song Gen Lyrics Formatter   : Format lyrics with section tags")
print("  - FL Song Gen Description Builder: Build style descriptions")
print("  - FL Song Gen Generate           : Generate songs with text conditioning")
print("  - FL Song Gen Style Transfer     : Generate songs with audio style reference")
print("  - FL Song Gen Auto Style         : Generate songs with preset styles")
print("  - FL Song Gen Audio Separator    : Separate audio into stems (vocals, drums, bass)")
print("  - FL Song Gen Advanced Generate  : Generate with advanced sampling (top_p)")
print("=" * 85)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

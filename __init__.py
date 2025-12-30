"""
FL Song Gen - AI Song Generation for ComfyUI
Based on Tencent's SongGeneration (LeVo) model

Generate complete songs with vocals and instrumentals from lyrics!
"""

import sys
import os

# Add current directory to path for imports
current_dir = os.path.dirname(__file__)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Import nodes
from .fl_nodes.model_loader import FL_SongGen_ModelLoader
from .fl_nodes.lyrics_formatter import FL_SongGen_LyricsFormatter
from .fl_nodes.description_builder import FL_SongGen_DescriptionBuilder
from .fl_nodes.generate import FL_SongGen_Generate
from .fl_nodes.style_transfer import FL_SongGen_StyleTransfer
from .fl_nodes.auto_style import FL_SongGen_AutoStyle

# Node registration for ComfyUI
NODE_CLASS_MAPPINGS = {
    "FL_SongGen_ModelLoader": FL_SongGen_ModelLoader,
    "FL_SongGen_LyricsFormatter": FL_SongGen_LyricsFormatter,
    "FL_SongGen_DescriptionBuilder": FL_SongGen_DescriptionBuilder,
    "FL_SongGen_Generate": FL_SongGen_Generate,
    "FL_SongGen_StyleTransfer": FL_SongGen_StyleTransfer,
    "FL_SongGen_AutoStyle": FL_SongGen_AutoStyle,
}

# Display names for the UI
NODE_DISPLAY_NAME_MAPPINGS = {
    "FL_SongGen_ModelLoader": "FL Song Gen Model Loader",
    "FL_SongGen_LyricsFormatter": "FL Song Gen Lyrics Formatter",
    "FL_SongGen_DescriptionBuilder": "FL Song Gen Description Builder",
    "FL_SongGen_Generate": "FL Song Gen Generate",
    "FL_SongGen_StyleTransfer": "FL Song Gen Style Transfer",
    "FL_SongGen_AutoStyle": "FL Song Gen Auto Style",
}

# Version info
__version__ = "1.0.0"

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
print("  - FL Song Gen Model Loader      : Load the song generation model")
print("  - FL Song Gen Lyrics Formatter  : Format lyrics with section tags")
print("  - FL Song Gen Description Builder: Build style descriptions")
print("  - FL Song Gen Generate          : Generate songs with text conditioning")
print("  - FL Song Gen Style Transfer    : Generate songs with audio style reference")
print("  - FL Song Gen Auto Style        : Generate songs with preset styles")
print("=" * 85)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

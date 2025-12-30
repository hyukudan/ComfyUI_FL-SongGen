# FL Song Gen

AI-powered song generation nodes for ComfyUI based on Tencent's SongGeneration (LeVo) model. Generate complete songs with vocals and instrumentals from lyrics!

[![SongGeneration](https://img.shields.io/badge/SongGeneration-Original%20Repo-blue?style=for-the-badge&logo=github&logoColor=white)](https://github.com/AslpLab/SongGeneration)
[![Patreon](https://img.shields.io/badge/Patreon-Support%20Me-F96854?style=for-the-badge&logo=patreon&logoColor=white)](https://www.patreon.com/Machinedelusions)

## Features

- **Full Song Generation** - Generate complete songs with vocals and instrumentals
- **Dual-Track Output** - Get separate vocal and background music tracks
- **Lyrics-to-Song** - Structured lyrics with sections (verse, chorus, bridge, intro, outro)
- **Style Transfer** - Use reference audio to guide the musical style
- **Text Descriptions** - Control gender, timbre, genre, emotion, and BPM
- **Auto Style Presets** - Quick generation with Pop, Rock, Jazz, and more
- **Long-Form Generation** - Up to 4 minutes 30 seconds per song

## Nodes

| Node | Description |
|------|-------------|
| **Model Loader** | Load SongGeneration model with memory options |
| **Lyrics Formatter** | Build properly formatted lyrics from sections |
| **Description Builder** | Create style descriptions from components |
| **Generate** | Main generation with text conditioning |
| **Style Transfer** | Generate using reference audio for style |
| **Auto Style** | Generate with preset style prompts |

## Installation

### Prerequisites

1. Clone the SongGeneration repository to your ComfyUI parent directory:
```bash
cd /path/to/ComfyUI/..
git clone https://github.com/AslpLab/SongGeneration.git
```

2. Download model weights from HuggingFace:
```bash
# Download to ComfyUI/models/songgen/songgeneration_base_new/
# Required files: config.yaml, model.pt
```

### ComfyUI Manager
Search for "FL Song Gen" and install.

### Manual
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/filliptm/ComfyUI_FL-SongGen.git
cd ComfyUI_FL-SongGen
pip install -r requirements.txt
```

## Quick Start

1. Add **FL Song Gen Model Loader** and select model variant
2. Add **FL Song Gen Lyrics Formatter** to build your lyrics
3. Add **FL Song Gen Description Builder** for style (optional)
4. Connect to **FL Song Gen Generate** node
5. Connect outputs to audio save/preview nodes

## Models

| Model | Max Duration | VRAM | Languages |
|-------|-------------|------|-----------|
| songgeneration_base | 2m30s | 10-16GB | Chinese |
| songgeneration_base_new | 2m30s | 10-16GB | Chinese, English |
| songgeneration_base_full | 4m30s | 12-18GB | Chinese, English |
| songgeneration_large | 4m30s | 22-28GB | Chinese, English |

Models must be downloaded manually from [HuggingFace](https://huggingface.co/aslp-lab/SongGeneration) to `ComfyUI/models/songgen/`.

## Lyrics Format

Lyrics use section tags separated by ` ; ` with phrases separated by `.`:

```
[intro-short] ; [verse] First line.Second line.Third line ; [chorus] Chorus line one.Chorus line two ; [outro-short]
```

**Available Section Tags:**
- `[intro-short]`, `[intro-medium]`, `[intro-long]` - Instrumental intro
- `[verse]` - Verse section (requires lyrics)
- `[chorus]` - Chorus section (requires lyrics)
- `[bridge]` - Bridge section (requires lyrics)
- `[inst-short]`, `[inst-medium]`, `[inst-long]` - Instrumental break
- `[outro-short]`, `[outro-medium]`, `[outro-long]` - Instrumental outro

## Style Descriptions

Format: `"voice_type, timbre, genre, emotion, instruments, the bpm is X"`

Example: `"female, warm, pop, emotional, piano and drums, the bpm is 120"`

## Auto Style Presets

- Pop, R&B, Dance, Jazz, Folk, Rock
- Chinese Style, Chinese Tradition, Chinese Opera
- Metal, Reggae, Auto

## Requirements

- Python 3.10+
- CUDA 11.8+ (for GPU acceleration)
- 16GB RAM minimum (32GB+ recommended)
- NVIDIA GPU with 10-28GB VRAM (depends on model)

**Note:** CPU-only mode is supported but very slow. Mac MPS may have limited support.

## License

Apache 2.0

## Credits

Based on [SongGeneration (LeVo)](https://github.com/AslpLab/SongGeneration) by Tencent AI Lab.

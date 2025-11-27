# FreeDoM-Video

Authors: Kaliazin Nikolai, Meir Roketlishvili, Viacheslav Iablochnikov

**Training-Free Controllable Video Generation with Diffusion Models**

FreeDoM-Video is a course project for controllable video generation using CogVideoX diffusion models. It enables various conditioning modes (depth, sketch, grayscale, segmentation) without requiring additional training based on FreeDoM for images.

## Features

- **Multiple Generation Tasks**:
  - **Text-to-Video**: Generate videos from text prompts with frame-level control
  - **Image-to-Video**: Generate videos starting from a reference image with intermediate frame guidance
  - **Keyframe Interpolation**: Generate smooth video transitions between first and last keyframes

- **Flexible Conditioning**:
  - **Depth**: Use depth maps for structural guidance (via Depth-Anything-V2)
  - **Sketch/Scribble**: Guide generation with edge/sketch representations
  - **Grayscale**: Control with grayscale intensity patterns
  - **Segmentation**: Semantic segmentation-based guidance (via SegFormer)

## Installation

```bash
# Clone the repository
git clone https://github.com/ErrorZ403/FreeDoM-Video.git
cd FreeDoM-Video

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

```
FreeDoM-Video/
├── demo_inference.py              # Main demo script for all tasks
├── requirements.txt               # Python dependencies
├── cogvideox_interpolation/
│   ├── pipeline.py                # Keyframe interpolation pipeline
│   └── datasets.py                # Dataset utilities
├── pipelines/
│   ├── pipeline_cogvideox.py           # Text-to-video pipeline with FreeDoM guidance
│   ├── pipeline_cogvideox_image2video.py  # Image-to-video pipeline with FreeDoM guidance
│   └── utils/
│       ├── models.py              # CSD model and differentiable augmenters (for sketch)
│       ├── metrics_utils.py       # Evaluation metrics (computing metrics)
│       ├── CSD/                   # Contrastive Style Descriptor module (for sketch)
│       └── preprocessors/
│           ├── depth_preprocessor.py       # Depth estimation preprocessor
│           └── segmentation_preprocessor.py # Semantic segmentation preprocessor
├── experiments/
│   ├── infer.py                   # Batch inference for interpolation
│   ├── keyframe_cogx.py           # Keyframe-conditioned generation
│   └── text_cogvideox.py          # Text-to-video experiments
└── example/
    └── rhino/                     # Example input frames and prompt
```

## Usage

### Quick Start Demo

```bash
# Image-to-Video generation with depth guidance
python demo_inference.py --task image2video --loss depth --folder example/rhino

# Text-to-Video generation with sketch guidance
python demo_inference.py --task text2video --loss sketch --folder example/rhino

# Keyframe interpolation (no guidance loss needed)
python demo_inference.py --task interpolation --folder example/rhino
```

### Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--task` | str | required | Task type: `image2video`, `text2video`, or `interpolation` |
| `--loss` | str | None | Guidance loss: `depth`, `sketch`, `gray`, or `segmentation` |
| `--folder` | str | `example/rhino` | Input folder with frames and prompt.txt |
| `--output` | str | `results` | Output directory for generated videos |
| `--seed` | int | 42 | Random seed for reproducibility |
| `--device` | str | `cuda` | Device to run inference on |

### Input Folder Format

Your input folder should contain:
```
your_folder/
├── 00000.jpg #First frame for I2V and Interpolation
├── 000**.jpg #Any middle frames for additional guidance
├── 00048.jpg #Last frame for all tasks 
└── prompt.txt   # Text description of the video
```

### Data

We downloaded DAVIS dataset and collect 30 videos from it with more than 50 frames. You can use any video from DAVIS: https://davischallenge.org/davis2017/code.html

## Guidance Parameters

### Understanding the Guidance Schedule

- **`guidance_step`**: List of integers specifying how many gradient update iterations to perform at each diffusion timestep. Higher values = stronger guidance but slower inference.
  
- **`guidance_lr`**: Learning rate for gradient updates at each timestep. Controls the strength of guidance corrections.

- **`fixed_frames`**: List of frame indices (0-48) to apply guidance on. These frames will be matched to the conditioning signal.

- **`travel_time`**: Tuple `(start, end)` defining the timestep range for time-travel optimization, which improves temporal coherence by re-noising and re-denoising.

## Conditioning Types

### Depth (`--loss depth`)
Uses Depth-Anything-V2 to extract depth maps from reference frames. Good for preserving scene structure and object positions.

### Sketch (`--loss sketch`)
Applies Sobel edge detection to create sketch-like representations. Useful for stylized generation or when edge structure is important.

### Grayscale (`--loss gray`)
Converts frames to grayscale for intensity-based guidance. Good for matching lighting and contrast patterns.

### Segmentation (`--loss segmentation`)
Uses SegFormer for semantic segmentation masks. Ideal for maintaining object boundaries and class-specific regions.

## Pre-trained Models

The framework automatically downloads the following models from Hugging Face:

| Model | Purpose | Hugging Face ID |
|-------|---------|-----------------|
| CogVideoX-5b-I2V | Image-to-Video generation | `THUDM/CogVideoX-5b-I2V` |
| CogVideoX-5b | Text-to-Video generation | `zai-org/CogVideoX-5b` |
| CogVideoX-Interpolation | Keyframe interpolation | `feizhengcong/CogvideoX-Interpolation` |
| Depth-Anything-V2 | Depth estimation | `depth-anything/Depth-Anything-V2-Small-hf` |
| SegFormer | Semantic segmentation | `nvidia/segformer-b0-finetuned-ade-512-512` |

## Hardware requirements

The poject is computational expensive. For I2V and T2V generetation with FreeDoM guidance you need around 70GB of memory. Whole project was done with Nvidia A800 GPU.

## Based on

This project builds upon:
- [CogVideoX](https://github.com/THUDM/CogVideo) - Base video diffusion model
- [CSD](https://github.com/learn2phoenix/CSD) - Sketch network base
- [FreeDoM](https://github.com/vvictoryuki/FreeDoM) - Training-free guidance methodology
- [CogVideoX-Interpolation](https://github.com/feizc/CogvideX-Interpolation) - CogVideoX-Interpolation network


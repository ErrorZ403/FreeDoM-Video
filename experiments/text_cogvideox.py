import os
import torch
from ..pipelines.pipeline_cogvideox import CogVideoXPipeline
from diffusers.utils import export_to_gif, export_to_video, load_image, make_image_grid
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np

pipe = CogVideoXPipeline.from_pretrained(
    "zai-org/CogVideoX-5b",
    torch_dtype=torch.bfloat16
)
pipe = pipe.to("cuda")
pipe.transformer.enable_gradient_checkpointing()

pipe.vae.enable_tiling()
pipe.vae.enable_slicing()
pipe.vae.enable_gradient_checkpointing()


def read_video(dr):
    frames = []

    for f in sorted(os.listdir(dr)):
        if '.txt' in f or '.ipynb' in f: continue
        frames.append(Image.open(f"{dr}/{f}").resize((720, 480), Image.Resampling.LANCZOS))

    return frames

def parse_line_to_list(line: str) -> list:
    """
    parse_line_to_list("1,3,4") -> [1, 3, 4]
    parse_line_to_list("1-3,4") -> [1, 2, 3, 4]
    """
    result = []
    parts = line.split(',')
    
    for part in parts:
        part = part.strip()
        if '-' in part:
            start_str, end_str = part.split('-')
            start, end = int(start_str), int(end_str)
            result.extend(range(start, end + 1))
        else:
            result.append(int(part))
    
    return result

folder = 'data/DAVIS'

for p in sorted(os.listdir(folder)):
    video_path = f'{folder}/{p}'
    
    video = read_video(video_path)
    
    with open(f"{video_path}/prompt.txt", "r") as f:
        prompt = f.read().strip()
    
    guidance_lr = [3e0] * 50
    guidance_step = [0] * 3 + [3] * 17 + [1] * 30
    
    travel_time = (15, 20)
    fixed_frames = "12,24,36,48"
    fixed_frames_ = parse_line_to_list(fixed_frames)
    
    save_dir = f"results/t2i_5b_video_{p}_segmentation.gif"
    
    video = pipe(
        video=video,
        prompt=prompt,
        num_videos_per_prompt=1,
        num_inference_steps=50,
        num_frames=49,
        guidance_scale=6,
        generator=torch.Generator(device="cuda").manual_seed(42),
        fixed_frames=fixed_frames_,
        guidance_step=guidance_step,
        guidance_lr=guidance_lr,
        loss_fn="segmentation",
        additional_inputs={"style_image": None},
        travel_time=travel_time
    ).frames[0]
    
    export_to_gif(video, save_dir, fps=8)
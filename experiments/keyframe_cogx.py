import os
import torch
from ..pipelines.pipeline_cogvideox_image2video import CogVideoXImageToVideoPipeline
from diffusers.utils import export_to_gif, export_to_video, load_image, make_image_grid
from PIL import Image
import decord

pipe = CogVideoXImageToVideoPipeline.from_pretrained(
    "THUDM/CogVideoX-5b-I2V",
    torch_dtype=torch.bfloat16
)
pipe = pipe.to("cuda")
pipe.transformer.enable_gradient_checkpointing()

pipe.vae.enable_tiling()
pipe.vae.enable_slicing()
pipe.vae.enable_gradient_checkpointing()

def make_frames(path):
    frames = []
    vr = decord.VideoReader(path)
    for i in range(len(vr)):
        frames.append(Image.fromarray(vr[i].asnumpy()))
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
    init = Image.open(f"{folder}/{p}/00000.jpg")
    init = init.resize((720, 480), Image.Resampling.LANCZOS)
    inter = Image.open(f"{folder}/{p}/00024.jpg")
    inter = inter.resize((720, 480), Image.Resampling.LANCZOS)
    last = Image.open(f"{folder}/{p}/00048.jpg")
    last = last.resize((720, 480), Image.Resampling.LANCZOS)

    with open(f"{folder}/{p}/prompt.txt", "r") as f:
        prompt = f.read().strip()
    
    video_init = [None] * 49
    video_init[:1] = [init] * 1
    video_init[1:25] = [inter] * 24 
    video_init[25:] = [last] * 24
    
    seed = 0
    st = 1.0
    
    guidance_lr = [3e0] * 50
    guidance_step = [10] * 5 + [5] * 5 + [2] * 10 + [0] * 30 
    
    travel_time = (7, 20)
    fixed_frames = "24,48"
    fixed_frames_ = parse_line_to_list(fixed_frames)
    
    save_dir = f"results/{p}_depth.gif"
    video = pipe(
        prompt=prompt,
        video=video_init,
        num_videos_per_prompt=1,
        num_inference_steps=50,
        guidance_scale=6,
        generator=torch.Generator(device="cuda").manual_seed(seed),
        strength=st,
        fixed_frames=fixed_frames_,
        guidance_step=guidance_step,
        guidance_lr=guidance_lr,
        loss_fn="depth",
        travel_time=travel_time,
    ).frames[0]
    
    export_to_gif(video, save_dir, fps=8)
import torch 
from diffusers.utils import export_to_video, load_image 
from ..cogvideox_interpolation.pipeline import CogVideoXInterpolationPipeline 
import os

pipe = CogVideoXInterpolationPipeline.from_pretrained(
    'feizhengcong/CogvideoX-Interpolation',
    torch_dtype=torch.bfloat16
)

pipe.enable_sequential_cpu_offload()
pipe.vae.enable_tiling()
pipe.vae.enable_slicing()

folder = '../data/DAVIS'

for p in sorted(os.listdir(folder)):
    first_image = load_image(f"{folder}/{p}/00000.jpg")
    last_image = load_image(f"{folder}/{p}/00048.jpg")
    
    with open(f"{folder}/{p}/prompt.txt", "r") as f:
        prompt = f.read().strip()
    
    video = pipe(
        prompt=prompt,
        first_image=first_image,
        last_image=last_image,
        num_videos_per_prompt=50,
        num_inference_steps=50,
        num_frames=49,
        guidance_scale=6,
        generator=torch.Generator(device="cuda").manual_seed(42),
    )[0]
    export_to_video(video[0], "results/gen_"+p+".mp4", fps=8)

    
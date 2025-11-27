import os
import argparse
import torch
from cogvideox_interpolation.pipeline import CogVideoXInterpolationPipeline
from pipelines.pipeline_cogvideox import CogVideoXPipeline
from pipelines.pipeline_cogvideox_image2video import CogVideoXImageToVideoPipeline
from diffusers.utils import export_to_gif, export_to_video, load_image, make_image_grid
from PIL import Image
import decord

def load_pipe(task, device):
    if task == "image2video":
        pipe = CogVideoXImageToVideoPipeline.from_pretrained(
            "THUDM/CogVideoX-5b-I2V",
            torch_dtype=torch.bfloat16
        )

    elif task == "text2video":
        pipe = CogVideoXPipeline.from_pretrained(
            "zai-org/CogVideoX-5b",
            torch_dtype=torch.bfloat16
        )
    elif task == "interpolation":
        pipe = CogVideoXInterpolationPipeline.from_pretrained(
            'feizhengcong/CogvideoX-Interpolation',
            torch_dtype=torch.bfloat16
        )
    else:
        raise ValueError("Task type must be in ['image2video', 'text2video', 'interpolation']")

    pipe = pipe.to(device)
    pipe.transformer.enable_gradient_checkpointing()

    pipe.vae.enable_tiling()
    pipe.vae.enable_slicing()
    pipe.vae.enable_gradient_checkpointing()

    return pipe

def make_frames(path):
        frames = []
        vr = decord.VideoReader(path)
        for i in range(len(vr)):
            frames.append(Image.fromarray(vr[i].asnumpy()))
        return frames

def read_video(dr):
    frames = []

    for f in sorted(os.listdir(dr)):
        if '.txt' in f or '.ipynb' in f: continue
        frames.append(Image.open(f"{dr}/{f}").resize((720, 480), Image.Resampling.LANCZOS))

    return frames

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True, choices=["image2video", "text2video", "interpolation"])
    parser.add_argument("--loss", type=str)
    parser.add_argument("--folder", type=str, default="example/rhino")
    parser.add_argument("--output", type=str, default="results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    pipe = load_pipe(args.task, args.device)

    folder = args.folder
    seed = args.seed
    st = 1.0

    if args.task == "image2video":
        init = Image.open(f"{folder}/00000.jpg")
        init = init.resize((720, 480), Image.Resampling.LANCZOS)
        inter = Image.open(f"{folder}/00024.jpg")
        inter = inter.resize((720, 480), Image.Resampling.LANCZOS)
        last = Image.open(f"{folder}/00048.jpg")
        last = last.resize((720, 480), Image.Resampling.LANCZOS)

        with open(f"{folder}/prompt.txt", "r") as f:
            prompt = f.read().strip()

        video_init = [None] * 49
        video_init[:1] = [init] * 1
        video_init[1:25] = [inter] * 24 
        video_init[25:] = [last] * 24

        guidance_lr = [3e0] * 50
        guidance_step = [10] * 5 + [5] * 5 + [2] * 10 + [0] * 30 

        travel_time = (7, 20)
        fixed_frames_ = [24, 48]

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
            loss_fn=args.loss,
            travel_time=travel_time,
        ).frames[0]

    elif args.task == "text2video":
        video = read_video(folder)
        
        with open(f"{folder}/prompt.txt", "r") as f:
            prompt = f.read().strip()
        
        guidance_lr = [3e0] * 50
        guidance_step = [0] * 3 + [3] * 17 + [1] * 30
        
        travel_time = (15, 20)
        fixed_frames_ = [12, 24, 36, 48]
            
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
            loss_fn=args.loss,
            travel_time=travel_time
        ).frames[0]

    else:
        first_image = load_image(f"{folder}/00000.jpg")
        last_image = load_image(f"{folder}/00048.jpg")
        
        with open(f"{folder}/prompt.txt", "r") as f:
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
        
    save_dir = f"{args.output}/demo_rhino_{args.loss}_{args.task}.gif"
    export_to_gif(video, save_dir, fps=8)

if __name__ == "__main__":
    main()
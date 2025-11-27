import argparse
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import cv2
from PIL import Image
from torchvision import transforms
from torchvision.models import inception_v3
from scipy import linalg
import imageio
import os
import torchvision.transforms as T
from image_gen_aux import DepthPreprocessor, LineArtPreprocessor, SegmentationPreprocessor
from pipelines.utils.models import DifferentiableAugmenter
from huggingface_hub import hf_hub_download


def load_i3d_model(device='cuda'):
    model_path = hf_hub_download(
        repo_id="flateon/FVD-I3D-torchscript",
        filename="i3d_torchscript.pt"
    )
    model = torch.jit.load(model_path).to(device)
    model.eval()
    return model


def load_video(path):
    #print(path)
    path = Path(path)
    if path.suffix in ['.gif', '.mp4']:
        reader = imageio.get_reader(path)
        frames = [Image.fromarray(np.array(frame), 'RGB') for frame in reader]
        return np.stack(frames) if len(frames) > 0 else []
    elif path.is_dir():
        files = sorted(path.glob('*.jpg')) + sorted(path.glob('*.png'))
        frames = [Image.open(f).resize((720, 480), Image.Resampling.LANCZOS) for f in files]
        return np.stack(frames)
    raise ValueError(f"Unsupported format: {path}")


def preprocess_for_i3d(video):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    frames = [transform(frame) for frame in video]
    return torch.stack(frames).permute(1, 0, 2, 3).unsqueeze(0)


def calculate_fvd(real_videos, fake_videos, device='cuda'):
    i3d = load_i3d_model(device)
    
    def get_features(videos):
        all_features = []
        with torch.no_grad():
            for video in videos:
                video_tensor = preprocess_for_i3d(video).to(device)
                features = i3d(video_tensor)
                all_features.append(features.cpu().numpy().flatten())
        return np.array(all_features)
    
    real_features = get_features(real_videos)
    fake_features = get_features(fake_videos)
    
    mu_real = np.mean(real_features, axis=0)
    sigma_real = np.cov(real_features, rowvar=False)
    mu_fake = np.mean(fake_features, axis=0)
    sigma_fake = np.cov(fake_features, rowvar=False)
    
    diff = mu_real - mu_fake
    covmean, _ = linalg.sqrtm(sigma_real.dot(sigma_fake), disp=False)
    
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    fvd = diff.dot(diff) + np.trace(sigma_real + sigma_fake - 2 * covmean)
    return fvd


def calculate_psnr(gen_video, real_video):    
    min_frames = min(len(gen_video), len(real_video))
    gen_video = gen_video[:min_frames]
    real_video = real_video[:min_frames]
    
    h, w = min(gen_video.shape[1], real_video.shape[1]), min(gen_video.shape[2], real_video.shape[2])
    gen_video = np.array([cv2.resize(frame, (w, h)) for frame in gen_video])
    real_video = np.array([cv2.resize(frame, (w, h)) for frame in real_video])
    
    mse = np.mean((gen_video.astype(float) - real_video.astype(float)) ** 2)
    if mse == 0:
        return float('inf')
    
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr


def calculate_depth_metric(gen_depths, real_depths):    
    gen_depths = np.squeeze(gen_depths)
    real_depths = np.squeeze(real_depths)
    
    if len(gen_depths.shape) == 4 and gen_depths.shape[-1] == 3:
        gen_depths = gen_depths[..., 0]
    if len(real_depths.shape) == 4 and real_depths.shape[-1] == 3:
        real_depths = real_depths[..., 0]
    
    min_frames = min(len(gen_depths), len(real_depths))
    gen_depths = gen_depths[:min_frames].astype(float)
    real_depths = real_depths[:min_frames].astype(float)
    
    mse = np.mean((gen_depths - real_depths) ** 2)
    return mse
    

def calculate_segmentation_metric(gen_segs, real_segs):    
    min_frames = min(len(gen_segs), len(real_segs))
    gen_segs = gen_segs[:min_frames]
    real_segs = real_segs[:min_frames]
    
    intersection = np.sum((gen_segs == real_segs))
    union = gen_segs.size
    iou = intersection / union
    return iou


def calculate_canny_metric(gen_canny, real_canny):    
    gen_canny = np.squeeze(gen_canny)
    real_canny = np.squeeze(real_canny)
    
    min_frames = min(len(gen_canny), len(real_canny))
    gen_canny = gen_canny[:min_frames]
    real_canny = real_canny[:min_frames]
    
    if len(gen_canny.shape) == 4 and gen_canny.shape[-1] == 3:
        gen_canny = gen_canny[..., 0]
    if len(real_canny.shape) == 4 and real_canny.shape[-1] == 3:
        real_canny = real_canny[..., 0]
    
    gen_binary = (gen_canny > 0.5).astype(float)
    real_binary = (real_canny > 0.5).astype(float)
    
    intersection = np.sum((gen_binary == 1) & (real_binary == 1))
    union = np.sum((gen_binary == 1) | (real_binary == 1))
    
    iou = intersection / (union + 1e-8)
    return iou

def prepare_depth(video, depth_preprocessor):
    cond_video = []
    with torch.no_grad():
        for frame in video:
            frame = Image.fromarray(frame)
            cond_video.append(depth_preprocessor(frame, return_type="pt").cpu().numpy())

    return np.array(cond_video)

def prepare_segmentation(video, segmentation_preprocessor):
    with torch.no_grad():
        cond_video = [segmentation_preprocessor(Image.fromarray(frame), return_class_ids=True, return_type="pt").cpu().numpy() for frame in video]

    return np.array(cond_video)

def prepare_sketch(video, aug):
    to_tensor = T.ToTensor()
    cond_video = [to_tensor(frame).to('cuda', dtype=torch.float16) for frame in video]
    with torch.no_grad():
        cond_video = [aug(frame, mode='scribble').cpu().numpy() for frame in cond_video]

    return np.array(cond_video)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--fake_path_all", type=str, default="/path/to/generated/videos")
    parser.add_argument("--real_path", type=str, default="/path/to/real/videos")
    args = parser.parse_args()
    
    fake_path_all = args.fake_path_all
    real_path = args.real_path
    
    depth_preprocessor = DepthPreprocessor.from_pretrained("depth-anything/Depth-Anything-V2-Large-hf").to('cuda')
    segmentation_preprocessor = SegmentationPreprocessor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512").to('cuda')
    aug = DifferentiableAugmenter().to('cuda').requires_grad_(False)

    real_videos = []
    for p_real in sorted(os.listdir(real_path)):
        real_videos.append(load_video(f"{real_path}/{p_real}")[:49])

    all_dirs_with_results = [f'{fake_path_all}/{p}' for p in sorted(os.listdir(fake_path_all)) if os.path.isdir(os.path.join(fake_path_all, p))]

    for p_fake in all_dirs_with_results:
        if '.ipynb' in p_fake: continue
        print(f'Start {p_fake}')
        fake_videos = []
        for video in sorted(os.listdir(p_fake)):
            if '.ipynb' in video: continue
            fake_videos.append(load_video(f"{p_fake}/{video}"))
        
        if len(fake_videos) == 0: continue
        
        fvd = calculate_fvd(real_videos, fake_videos)
        print(f"FVD: {fvd:.4f}")

        psnr = 0.0
        depth_metric = 0.0
        seg_metric = 0.0
        canny_metric = 0.0
        for gen, real in zip(fake_videos, real_videos):
            psnr += calculate_psnr(gen, real)
            
            gen_depths = prepare_depth(gen, depth_preprocessor)
            real_depths = prepare_depth(real, depth_preprocessor)
            depth_metric += calculate_depth_metric(gen_depths, real_depths)

            gen_segs = prepare_segmentation(gen, segmentation_preprocessor)
            real_segs = prepare_segmentation(real, segmentation_preprocessor)
            seg_metric += calculate_segmentation_metric(gen_segs, real_segs)

            gen_canny = prepare_sketch(gen, aug)
            real_canny = prepare_sketch(real, aug)

            canny_metric += calculate_canny_metric(gen_canny, real_canny)
        
        psnr /= len(fake_videos)
        print(f"PSNR: {psnr:.2f} dB")

        depth_metric /= len(fake_videos)
        print(f"Depth Abs Rel Error: {depth_metric:.4f}")

        seg_metric /= len(fake_videos)
        print(f"Segmentation IoU: {seg_metric:.4f}")
        
        canny_metric /= len(fake_videos)
        print(f"Canny F1 Score: {canny_metric:.4f}")
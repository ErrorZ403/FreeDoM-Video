import os
from typing import List, Union, Dict, Sequence, Optional

import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
from transformers import AutoModelForSemanticSegmentation

from .utils_preproc import ImageMixin
from .utils_preproc import Preprocessor


class SegmentationPreprocessor(Preprocessor, ImageMixin):
    """
    Preprocessor for semantic segmentation with Transformers models.

    - Loads any AutoModelForSemanticSegmentation.
    - Freezes weights.
    - Accepts PIL / NumPy / torch images or lists thereof.
    - Returns either:
        • grayscale class-id masks normalized to [0,1] (default), or
        • colorized RGB masks if `colorize=True`.
    - Keeps resolution consistent with input; will upsample model logits if needed.
    """

    @classmethod
    def from_pretrained(cls, pretrained_model_or_path: Union[str, os.PathLike]):
        model = AutoModelForSemanticSegmentation.from_pretrained(pretrained_model_or_path)
        for p in model.parameters():
            p.requires_grad = False
        return cls(model)

    def __call__(
        self,
        image: Union[PIL.Image.Image, np.ndarray, torch.Tensor, List[PIL.Image.Image]],
        resolution_scale: float = 1.0,
        batch_size: int = 1,
        return_type: str = "pil",
        *,
        colorize: bool = False,
        palette: Optional[Sequence[Sequence[int]]] = None,  # e.g. [(0,0,0), (128,0,0), ...]
        id2label: Optional[Dict[int, str]] = None,
        return_class_ids: bool = False,  # if True, bypass normalization/post and return raw class id tensor
        return_logits: bool = False
    ):
        """
        Args:
            image: Input image(s).
            resolution_scale: Pre-scale factor before inference (like your depth code).
            batch_size: Mini-batch size.
            return_type: Passed to ImageMixin.post_process_image.
            colorize: If True, output RGB masks using `palette` (or an auto palette).
            palette: Optional list of RGB triples for classes; length should cover num_classes.
            id2label: Optional class name mapping (not required, but kept for parity).
            return_class_ids: If True, returns integer class-id tensor(s) without normalization.
        """
        if not isinstance(image, torch.Tensor):
            image = self.convert_image_to_tensor(image).to(self.model.device)

        image, resolution_scale = self.scale_image(image, resolution_scale)

        processed_batches = []
        class_ids_batches = []
        logits_batches = []

        inferred_num_classes: Optional[int] = None

        for i in range(0, len(image), batch_size):
            batch = image[i : i + batch_size].to(self.model.device)  # [B,3,H,W]
            outputs = self.model(batch)
            logits: torch.Tensor = outputs.logits  # [B,C,h,w]

            if (logits.shape[-2] != batch.shape[-2]) or (logits.shape[-1] != batch.shape[-1]):
                logits = F.interpolate(logits, size=(batch.shape[-2], batch.shape[-1]),
                                        mode="bilinear", align_corners=False)

            inferred_num_classes = logits.shape[1]

            if return_logits:
                logits_batches.append(logits)
            else:
                class_ids = logits.softmax(dim=1).argmax(dim=1, keepdim=True)  # [B,1,H,W]
                class_ids_batches.append(class_ids)

        if return_logits:
            return torch.cat(logits_batches, dim=0)

        class_ids_full = torch.cat(class_ids_batches, dim=0)  # [N,1,H,W], int64

        if return_class_ids:
            if resolution_scale != 1.0:
                class_ids_full = F.interpolate(class_ids_full.float(),
                                               scale_factor=1.0 / resolution_scale,
                                               mode="nearest").long()
            return class_ids_full 
        if colorize:
            num_classes = inferred_num_classes or int(class_ids_full.max().item() + 1)
            if palette is None or len(palette) < num_classes:
                rng = torch.Generator().manual_seed(0)
                auto = torch.randint(0, 256, (num_classes, 3), generator=rng, dtype=torch.int64)
                if palette is not None:
                    given = torch.tensor(palette, dtype=torch.int64)
                    auto[: len(given)] = given
                palette_tensor = auto  # [C,3]
            else:
                palette_tensor = torch.tensor(palette, dtype=torch.int64)  # [C,3]

            palette_tensor = palette_tensor.to(torch.int64)
            flat = class_ids_full.squeeze(1).to(torch.int64)  # [N,H,W]
            rgb = palette_tensor[flat]  # [N,H,W,3]
            rgb = rgb.permute(0, 3, 1, 2).contiguous()  # [N,3,H,W]
            rgb = rgb.to(torch.float32) / 255.0

            processed = rgb
        else:
            num_classes = (inferred_num_classes or int(class_ids_full.max().item() + 1))
            denom = max(num_classes - 1, 1)
            processed = class_ids_full.to(torch.float32) / float(denom)  # [N,1,H,W]
            processed = processed.clamp(0, 1)

        if resolution_scale != 1.0:
            interp_mode = "nearest" if not colorize else "bilinear"
            processed = F.interpolate(
                processed,
                scale_factor=1.0 / resolution_scale,
                mode=interp_mode,
                align_corners=False if interp_mode == "bilinear" else None,
            )

        image_out = self.post_process_image(processed, return_type)

        return image_out
import io

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import torch
from PIL import Image
from transformers import (
    AutoImageProcessor,
    AutoModel,
    AutoProcessor,
    CLIPImageProcessor,
    CLIPVisionModel,
    IJepaModel,
)

from . import config


def get_vision_model(vmodel: str, device: str = "cuda"):
    """Load a vision encoder + its image processor by short name.

    Returns (model, image_processor). The model is placed on `device`.
    """
    cfg = config.get_vision_config(vmodel)
    hf = cfg["hf"]

    if vmodel == "clip":
        model = CLIPVisionModel.from_pretrained(hf).to(device)
        processor = CLIPImageProcessor.from_pretrained(hf)
    elif vmodel == "dinov2":
        model = AutoModel.from_pretrained(hf).to(device)
        processor = AutoImageProcessor.from_pretrained(hf)
    elif vmodel == "jepa":
        model = IJepaModel.from_pretrained(hf, device_map=device)
        processor = AutoProcessor.from_pretrained(hf)
    else:  # should be unreachable because get_vision_config validates
        raise ValueError(f"Unknown vision model {vmodel!r}")

    return model, processor


def transform_bbox_to_crop_space(bbox, original_size, vmodel="dinov2"):
    """Transform [x, y, w, h] from original image space to model crop space.

    DINOv2: shortest edge -> 256, center crop 224.
    CLIP (standard) / JEPA: shortest edge -> 224, center crop 224.
    CLIP-336: shortest edge -> 336, center crop 336.
    """
    assert vmodel in ("dinov2", "jepa", "clip", "clip336")
    orig_w, orig_h = original_size
    x, y, w, h = bbox

    if vmodel.startswith("dinov2"):
        shortest_edge_target = 256
        crop_size = 224
    elif vmodel == "clip336":
        shortest_edge_target = 336
        crop_size = 336
    else:
        shortest_edge_target = 224
        crop_size = 224

    scale = shortest_edge_target / min(orig_w, orig_h)
    offset_x = (orig_w * scale - crop_size) / 2
    offset_y = (orig_h * scale - crop_size) / 2

    nx1 = (x * scale) - offset_x
    ny1 = (y * scale) - offset_y
    nx2 = nx1 + w * scale
    ny2 = ny1 + h * scale

    final_x1 = max(0, min(crop_size, nx1))
    final_y1 = max(0, min(crop_size, ny1))
    final_x2 = max(0, min(crop_size, nx2))
    final_y2 = max(0, min(crop_size, ny2))

    return [final_x1, final_y1, final_x2 - final_x1, final_y2 - final_y1]


_PATCH_CONFIG = {
    "clip":   {"shortest_edge": 224, "resample": Image.Resampling.BICUBIC},
    "dinov2": {"shortest_edge": 256, "resample": Image.Resampling.BICUBIC},
    "jepa":   {"shortest_edge": 224, "resample": Image.Resampling.BILINEAR},
}


def plot_model_patch_grid(pil_image, model_type="clip", highlight_patches=None,
                          include_grid=False, alpha=0.6, facecolor="yellow"):
    """Render the 14x14 patch grid that the encoder will see, highlighting patches."""
    assert model_type in _PATCH_CONFIG, f"Unknown model_type: {model_type}"
    params = _PATCH_CONFIG[model_type]
    shortest_edge = params["shortest_edge"]
    resample_method = params["resample"]

    patch_size = 14
    crop_size = 224

    # Mimic ImageProcessor resize
    w, h = pil_image.size
    scale = shortest_edge / min(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized_image = pil_image.resize((new_w, new_h), resample=resample_method)

    # Mimic center crop
    left = (new_w - crop_size) / 2
    top = (new_h - crop_size) / 2
    cropped_image = resized_image.crop((left, top, left + crop_size, top + crop_size))

    fig, ax = plt.subplots(1, figsize=(8, 8))
    ax.imshow(cropped_image)

    num_patches_per_row = crop_size // patch_size
    highlight_set = set(highlight_patches) if highlight_patches else set()

    for row in range(num_patches_per_row):
        for col in range(num_patches_per_row):
            patch_index = row * num_patches_per_row + col
            x, y = col * patch_size, row * patch_size

            if patch_index in highlight_set:
                rect = patches.Rectangle(
                    (x, y), patch_size, patch_size,
                    linewidth=0, facecolor=facecolor, alpha=alpha,
                )
                ax.add_patch(rect)

            if include_grid:
                ax.text(
                    x + patch_size / 2, y + patch_size / 2, str(patch_index),
                    color="lime", fontsize=6, ha="center", va="center", weight="bold",
                )

    ax.axis("off")
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0, dpi=300)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


class VisionPatchMapper:
    """Maps bounding boxes to the patches a ViT-style encoder would see.

    Correctly handles DINOv2's 256-resize vs 224-crop discrepancy and
    CLIP-336's 336-resize vs 336-crop.
    """

    def __init__(self, vision_model, image_processor):
        self.patch_size = vision_model.config.patch_size

        # Prefer crop_size (the actual tensor shape fed to the model);
        # fall back to size if crop_size is missing.
        if hasattr(image_processor, "crop_size"):
            cs = image_processor.crop_size
            self.image_size = cs.get("height", 224) if isinstance(cs, dict) else cs
        elif hasattr(image_processor, "size"):
            sz = image_processor.size
            if isinstance(sz, dict):
                self.image_size = (
                    sz.get("height") or sz.get("shortest_edge") or 224
                )
            else:
                self.image_size = sz
        else:
            self.image_size = 224

        print(
            f"VisionPatchMapper: input={self.image_size}x{self.image_size}, "
            f"patch={self.patch_size}, grid={self.image_size // self.patch_size}"
        )

    def get_patch_grid(self):
        n = self.image_size // self.patch_size
        patches_ = []
        for i in range(n):
            for j in range(n):
                x1, y1 = j * self.patch_size, i * self.patch_size
                patches_.append({
                    "patch_idx": len(patches_),
                    "coords": (x1, y1, x1 + self.patch_size, y1 + self.patch_size),
                    "grid_pos": (i, j),
                })
        return {"patches": patches_, "grid_shape": (n, n), "total_patches": len(patches_)}

    def bbox_to_patches(self, bbox, patch_grid):
        x1, y1, w, h = bbox
        x2, y2 = x1 + w, y1 + h
        intersecting = []
        for p in patch_grid["patches"]:
            px1, py1, px2, py2 = p["coords"]
            ix = max(0, min(x2, px2) - max(x1, px1))
            iy = max(0, min(y2, py2) - max(y1, py1))
            if ix > 0 and iy > 0:
                intersecting.append({
                    "patch_idx": p["patch_idx"],
                    "overlap_ratio": (ix * iy) / (self.patch_size ** 2),
                })
        return {"intersecting_patches": intersecting}

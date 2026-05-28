import gc
import os
from pathlib import Path

import torch
from pycocotools.coco import COCO
from tqdm import tqdm

from . import config


# ---------------------------------------------------------------------------
# COCO helpers
# ---------------------------------------------------------------------------
def extract_coco_classes(coco_path, classes=("dog", "cat"), output_dir="./extracted",
                          max_results=500, splits=("val",)):
    """Extract images containing specific classes from COCO."""
    os.makedirs(output_dir, exist_ok=True)
    ann_train = os.path.join(coco_path, "annotations", "instances_train2017.json")
    ann_val = os.path.join(coco_path, "annotations", "instances_val2017.json")

    print("Loading COCO annotations...")
    coco_train = COCO(ann_train) if "train" in splits else None
    coco_val = COCO(ann_val) if "val" in splits else None

    # Get class IDs from whichever coco we have loaded
    ref = coco_train or coco_val
    class_ids = {}
    for cn in classes:
        ids = ref.getCatIds(catNms=[cn])
        if ids:
            class_ids[cn] = ids[0]
            print(f"Found class {cn!r} with ID: {ids[0]}")
        else:
            print(f"Class {cn!r} not found in COCO")
    if not class_ids:
        print("No target classes found!")
        return {}, ref

    results = {}
    splits_map = {"train": coco_train, "val": coco_val}
    for split_name in splits:
        coco = splits_map[split_name]
        if coco is None:
            continue
        print(f"\nProcessing {split_name} split...")

        img_ids = set()
        for cat_id in class_ids.values():
            img_ids.update(coco.getImgIds(catIds=[cat_id]))
        print(f"Found {len(img_ids)} images containing target classes")

        for i, img_id in enumerate(list(img_ids)[:max_results]):
            if i % 100 == 0:
                print(f"  Processing image {i}/{len(img_ids)}...")
            img_info = coco.loadImgs([img_id])[0]
            ann_ids = coco.getAnnIds(imgIds=[img_id])
            annotations = coco.loadAnns(ann_ids)

            target_anns = [a for a in annotations if a["category_id"] in class_ids.values()]
            if not target_anns:
                continue

            key = f"{split_name}_{img_id}"
            classes_present = []
            for ann in target_anns:
                cat_info = coco.loadCats([ann["category_id"]])[0]
                if cat_info["name"] not in classes_present:
                    classes_present.append(cat_info["name"])

            results[key] = {
                "image_info": img_info,
                "image_path": os.path.join(coco_path, f"{split_name}2017", img_info["file_name"]),
                "annotations": target_anns,
                "classes_present": classes_present,
            }

    # Return the train coco for backward compat with the original signature
    return results, (coco_train if coco_train else coco_val)


def extract_coco_by_ids(coco_path, image_ids, splits=("val",)):
    """Extract images and annotations from COCO given specific image IDs."""
    coco_objects = {}
    if "train" in splits:
        ann = os.path.join(coco_path, "annotations", "instances_train2017.json")
        print("Loading Train annotations...")
        coco_objects["train"] = COCO(ann)
    if "val" in splits:
        ann = os.path.join(coco_path, "annotations", "instances_val2017.json")
        print("Loading Val annotations...")
        coco_objects["val"] = COCO(ann)

    results = {}
    for split_name, coco in coco_objects.items():
        print(f"\nSearching for IDs in {split_name} split...")
        valid_ids = coco.getImgIds(imgIds=list(image_ids))
        print(f"Found {len(valid_ids)} of {len(image_ids)} IDs in {split_name}")

        for i, img_id in enumerate(valid_ids):
            if i % 100 == 0 and i > 0:
                print(f"  Processing image {i}/{len(valid_ids)}...")
            img_info = coco.loadImgs([img_id])[0]
            ann_ids = coco.getAnnIds(imgIds=[img_id])
            annotations = coco.loadAnns(ann_ids)

            key = f"{split_name}_{img_id}"
            classes_present = []
            for ann in annotations:
                cat_info = coco.loadCats([ann["category_id"]])[0]
                if cat_info["name"] not in classes_present:
                    classes_present.append(cat_info["name"])

            results[key] = {
                "image_info": img_info,
                "image_path": os.path.join(coco_path, f"{split_name}2017", img_info["file_name"]),
                "annotations": annotations,
                "classes_present": classes_present,
            }
    return results


# ---------------------------------------------------------------------------
# Cached activation loaders
# ---------------------------------------------------------------------------
# Maps (model_type, computation) -> the key inside the cached example_*.pt file.
# Cached files are saved by scripts/cache_activations.py under example_data[f'layer_{i}'].
_ACT_KEYS = {
    # raw (post-block) activations
    ("vlm_img",     None): "image_activations",
    ("vlm_imgtext", None): "image_text_activations",
    ("llm_text",    None): "text_activations",
    # SAE-encoded
    ("vlm_img",     "sae_encoded"): "sae_image_activations",
    ("vlm_imgtext", "sae_encoded"): "sae_image_text_activations",
    ("llm_text",    "sae_encoded"): "sae_text_activations",
    # SAE-decoded (reconstructions)
    ("vlm_img",     "sae_decoded"): "sae_image_reconstructions",
    ("vlm_imgtext", "sae_decoded"): "sae_image_text_reconstructions",
    ("llm_text",    "sae_decoded"): "sae_text_reconstructions",
}


def _example_path(base_dir, idx):
    return Path(base_dir) / f"example_{idx}.pt"



def load_activations(
    computation=None,
    components=None,
    layers=None,
    indices=None,
    base_dir='activations',
    model_type='vlm_imgtext'
):
    base_path = Path(base_dir)
    
    # Handle SAE computations (encoded/decoded)
    if computation in ['sae_encoded', 'sae_decoded']:
        result = {layer: [] for layer in layers}
        
        for idx in indices:
            file_path = base_path / f'example_{idx}.pt'
            if not file_path.exists():
                print(f"Warning: File not found for example {idx}")
                continue
                
            data = torch.load(file_path, weights_only=False)
            
            for layer in layers:
                if layer >= len(data):
                    print(f"Warning: Layer {layer} not found in example {idx}")
                    continue
                
                # Extract the appropriate activations based on computation type
                if computation == 'sae_encoded':
                    if model_type == 'vlm_img':
                        acts = data[f'layer_{layer}']['sae_image_activations']
                    elif model_type == 'vlm_imgtext':
                        acts = data[f'layer_{layer}']['sae_image_text_activations']
                    elif model_type == 'llm_text':
                        acts = data[f'layer_{layer}']['sae_text_activations']
                else:  # sae_decoded
                    if model_type == 'vlm_img':
                        acts = data[f'layer_{layer}']['sae_image_reconstructions']
                    elif model_type == 'vlm_imgtext':
                        acts = data[f'layer_{layer}']['sae_image_text_reconstructions']
                    elif model_type == 'llm_text':
                        acts = data[f'layer_{layer}']['sae_text_reconstructions']

                result[layer].append(acts.to("cuda"))
        
        return result

    elif computation == 'projector_outputs':

        results = []
        for idx in indices:
            file_path = base_path / f'example_{idx}.pt'
            if not file_path.exists():
                print(f"Warning: File not found for example {idx}")
                continue
            
            data = torch.load(file_path, weights_only=False)
            results.append(data['projector_outputs'].to("cuda"))
        
        return results
    
    # Handle raw activations
    else:
        result = {comp: {layer: [] for layer in layers} for comp in components}
        
        for idx in indices:
            file_path = base_path / f'example_{idx}.pt'
            if not file_path.exists():
                print(f"Warning: File not found for example {idx}")
                continue
                
            data = torch.load(file_path, weights_only=False)
            
            for comp in components:
                for layer in layers:
                    if layer >= len(data):
                        print(f"Warning: Layer {layer} not found in example {idx}")
                        continue
                    
                    # For raw activations, use the image_activations as layer output
                    if model_type == 'vlm_img':
                        acts = data[f'layer_{layer}']['image_activations']
                    elif model_type == 'vlm_imgtext':
                        acts = data[f'layer_{layer}']['image_text_activations']
                    elif model_type == 'llm_text':
                        acts = data[f'layer_{layer}']['text_activations']
                    result[comp][layer].append(acts.to("cuda"))
        
        return result



def load_activations_batch(computation, layers, start_idx, end_idx,
                           model_type, base_dir=None):
    """Convenience wrapper for sequential index ranges."""
    if base_dir is None:
        base_dir = config.ACTIVATIONS_DIR

    indices = list(range(start_idx, end_idx))
    if computation is None:
        return load_activations(
            components=["layer_output"], layers=layers, indices=indices,
            model_type=model_type, base_dir=base_dir,
        )["layer_output"]
    return load_activations(
        computation=computation, layers=layers, indices=indices,
        model_type=model_type, base_dir=base_dir,
    )
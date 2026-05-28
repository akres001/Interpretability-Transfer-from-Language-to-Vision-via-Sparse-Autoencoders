import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from vlm_sae import config
from vlm_sae.api import LEXICAL_AUDITOR_PROMPT, process_loop
from vlm_sae.data import extract_coco_by_ids
from vlm_sae.vision import (
    VisionPatchMapper,
    get_vision_model,
    plot_model_patch_grid,
    transform_bbox_to_crop_space,
)


def collect_matched_results(detailed_results, token_type="vlm_img"):
    """Pull (layer, feature, token, image, description) tuples out of matching_rate output."""
    matched = []
    for entry in detailed_results["evaluations"]:
        if entry["token_type"] != token_type:
            continue
        if entry.get("match_found") != 1:
            continue
        for match_id in entry["feature_ids_match"]:
            idx = match_id - 1
            try:
                if entry["act_values"][idx] == 0:
                    continue
                explanation = next(
                    m["explanation"]
                    for m in entry["explanation"]["matches"]
                    if m["feature_id"] == match_id
                )
                matched.append({
                    "layer": entry["layer"],
                    "feature_index": entry["feature_indices"][idx],
                    "act_value": entry["act_values"][idx],
                    "token_idxs": entry["token_idxs"][idx],
                    "image_id": entry["image_id"],
                    "unique_id": entry["unique_id"],
                    "description": entry["feature_descriptions"][idx],
                    "explanation": explanation,
                })
            except (IndexError, StopIteration, KeyError):
                continue
    return matched


def build_valid_image_index(coco_results, coco, vision_model_name, mapper,
                             min_patches=5, max_patches=200):
    """Build a {f'{split}_{img_id}_{class_name}': (pil, path, patches, [class])} mapping.

    Groups annotations of the same class within an image into a single set of
    patches (union over bboxes).
    """
    valid = {}
    for img_key in tqdm(coco_results, desc="Mapping bboxes -> patches"):
        data = coco_results[img_key]
        img = Image.open(data["image_path"])
        for ann in data["annotations"]:
            cat_info = coco.loadCats([ann["category_id"]])[0]
            class_name = cat_info["name"]
            group_key = f"{img_key}_{class_name}"

            bbox = transform_bbox_to_crop_space(ann["bbox"], img.size, vmodel=vision_model_name)
            patch_grid = mapper.get_patch_grid()
            mapping = mapper.bbox_to_patches(bbox, patch_grid)
            patch_indices = sorted(p["patch_idx"] for p in mapping["intersecting_patches"])
            if not (min_patches <= len(patch_indices) <= max_patches):
                continue

            if group_key in valid:
                _, path, existing, classes = valid[group_key]
                merged = list(set(existing + patch_indices))
                valid[group_key] = (None, path, merged, classes)
            else:
                valid[group_key] = (None, data["image_path"], patch_indices, [class_name])

    # Render the grids once at the end, after all patches are merged
    print("Rendering visualization grids...")
    for group_key in tqdm(valid):
        _, img_path, all_patches, all_classes = valid[group_key]
        img = Image.open(img_path)
        img_pil = plot_model_patch_grid(
            img, model_type=vision_model_name,
            highlight_patches=all_patches, include_grid=True,
        )
        valid[group_key] = (img_pil, img_path, all_patches, all_classes)
    return valid


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--matching-results", required=True,
                    help="JSON output from matching_rate.py")
    p.add_argument("--vision-model", default="dinov2",
                    choices=["clip", "dinov2", "jepa"])
    p.add_argument("--coco-path", default=str(config.COCO_PATH))
    p.add_argument("--splits", nargs="+", default=["train"])
    p.add_argument("--output",
                    default=str(config.RESULTS_DIR / "localization_results.json"))
    p.add_argument("--auditor-model", default="gemini-3-flash-preview")
    p.add_argument("--audit-cache", default=None,
                    help="Path to cache the raw lexical-auditor responses. "
                         "Auto-named next to --output if unset.")
    # p.add_argument("--recompute-audits", action="store_true",
                    # help="Re-run the auditor even if a cached file exists.")
    return p.parse_args()


def main():
    args = parse_args()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    detailed_results = json.loads(Path(args.matching_results).read_text())
    matched_results = collect_matched_results(detailed_results, token_type="vlm_img")
    print(f"{len(matched_results)} matched (feature, image) pairs to evaluate")

    # --- COCO lookup for the bboxes of the matched images ---
    image_ids = [
        int(el["image_id"].split("/")[-1].replace(".jpg", ""))
        for el in matched_results
    ]
    coco_results = extract_coco_by_ids(args.coco_path, image_ids, splits=args.splits)

    # --- Mapper ---
    vision_model, processor = get_vision_model(args.vision_model, device="cpu")
    mapper = VisionPatchMapper(vision_model, processor)

    # We need a COCO object for category lookups. Reuse one of the loaded splits.
    from pycocotools.coco import COCO
    split = args.splits[0]
    coco = COCO(str(Path(args.coco_path) / "annotations" / f"instances_{split}2017.json"))

    valid_images = build_valid_image_index(coco_results, coco, args.vision_model, mapper)

    # --- Build the lexical-audit batch ---
    all_data_list = []
    prompt_counter = 0
    for el in matched_results:
        imid = int(el["image_id"].split("/")[-1].replace(".jpg", ""))
        for split in args.splits:
            base_key = f"{split}_{imid}"
            associated_keys = [k for k in valid_images if k.startswith(f"{base_key}_")]
            for full_key in associated_keys:
                key_word = valid_images[full_key][-1][0]
                concept = el["description"]
                all_data_list.append({
                    "request_id": f"request_{prompt_counter}",
                    "prompt": LEXICAL_AUDITOR_PROMPT.format(
                        keyword=key_word, concept_description=concept,
                    ),
                })
                prompt_counter += 1
    print(f"Built {len(all_data_list)} audit prompts")

    # --- Run audits (sequential with backoff). Use upload_batch if you have many. ---
    audit_cache_path = Path(args.audit_cache) if args.audit_cache else (
        Path(args.output).with_name(Path(args.output).stem + "_audit_responses.json")
    )
    if audit_cache_path.exists() and not args.recompute_audits:
        print(f"Loading cached audit responses from {audit_cache_path}")
        audit_responses = json.loads(audit_cache_path.read_text())
    else:
        audit_responses = process_loop(all_data_list, model=args.auditor_model)
        audit_cache_path.parent.mkdir(parents=True, exist_ok=True)
        audit_cache_path.write_text(json.dumps(audit_responses, indent=2, default=str))
        print(f"Saved {len(audit_responses)} audit responses to {audit_cache_path}")

    # --- Pair audits back to matched results ---
    outputs_full_raw = {}
    for req_id, raw_text in audit_responses.items():
        try:
            json.loads(raw_text)
            outputs_full_raw[req_id] = raw_text
        except Exception:
            pass

    matched_results_selected = []
    retrieval_counter = 0
    for el in matched_results:
        imid = int(el["image_id"].split("/")[-1].replace(".jpg", ""))
        for split in args.splits:
            base_key = f"{split}_{imid}"
            associated_keys = [k for k in valid_images if k.startswith(f"{base_key}_")]
            for full_key in associated_keys:
                req_id = f"request_{retrieval_counter}"
                retrieval_counter += 1
                if req_id not in outputs_full_raw:
                    continue
                key_word = valid_images[full_key][-1][0]
                new_el = dict(el)
                new_el["match_concept"] = outputs_full_raw[req_id]
                new_el["key_word"] = key_word
                new_el["valid_image_key"] = full_key
                new_el["corresponding_patches"] = valid_images[full_key][2]
                matched_results_selected.append(new_el)
    
    # --- Filter to only the cases where the auditor said "yes, the keyword is present" ---
    final_selections = []
    for el in matched_results_selected:
        try:
            raw = el["match_concept"].replace("```json", "").replace("```", "").strip()
            data = json.loads(raw)
            # Some auditor responses come back as a single-element list rather than an object
            if isinstance(data, list):
                data = data[0] if data else {}
            if not isinstance(data, dict):
                continue
            if data.get("match") == 1:
                el["parsed_analysis"] = data
                final_selections.append(el)
        except (json.JSONDecodeError, KeyError, TypeError, AttributeError, IndexError):
            continue
    print(f"Total valid matches after auditor: {len(final_selections)}")

    # --- Localization accuracy ---
    matches = np.array([
        1 if el["token_idxs"] in el["corresponding_patches"] else 0
        for el in final_selections
    ])
    acc = float(matches.mean()) if matches.size else 0.0
    print(f"Localization accuracy: {acc:.4f} ({matches.sum()}/{matches.size})")

    Path(args.output).write_text(json.dumps({
        "accuracy": acc,
        "n_total": int(matches.size),
        "n_correct": int(matches.sum()),
        "selections": final_selections,
    }, indent=2, default=str))
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
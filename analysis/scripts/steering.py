import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

from vlm_sae import config
from vlm_sae.api import (
    evaluate_interpretability,
    get_feature_information,
)
from vlm_sae.data import extract_coco_classes
from vlm_sae.models import LLaVAModel, load_language_model, load_saes
from vlm_sae.vision import (
    VisionPatchMapper,
    get_vision_model,
    plot_model_patch_grid,
    transform_bbox_to_crop_space,
)


LATENTS_BY_MODEL = {
    "gemma-2-2b-it": {
        "cat":   {"feat": 15456, "layer": 4},
        "dog":   {"feat": 1008,  "layer": 13},
        "bird":  {"feat": 6984,  "layer": 23},
        "cow":   {"feat": 9096,  "layer": 22},
        "pizza": {"feat": 16100, "layer": 22},
        "bus":   {"feat": 15576, "layer": 23},
        "meat steak or related": {"feat": 5329, "layer": 18},
    },
    "gemma-2-9b-it": {
        "cat":   {"feat": 6897,  "layer": 33},
        "dog":   {"feat": 9800,  "layer": 19},
        "bird":  {"feat": 7819,  "layer": 7},
        "cow":   {"feat": 16247, "layer": 6},
        "pizza": {"feat": 15533, "layer": 8},
        "bus":   {"feat": 8412,  "layer": 32},
        "meat steak or related": {"feat": 15130, "layer": 37},
    },
    "meta-llama/Llama-3.1-8B-Instruct": {
        "cat":   {"feat": 12216, "layer": 20},
        "dog":   {"feat": 20111, "layer": 6},
        "bird":  {"feat": 2392,  "layer": 27},
        "cow":   {"feat": 5040,  "layer": 28},
        "pizza": {"feat": 11563, "layer": 23},
        "bus":   {"feat": 10801, "layer": 29},
        "meat steak or related": {"feat": 15198, "layer": 21},
    },
}

CLASS_MAP = {
    "cat": "dog",
    "dog": "cat",
    "pizza": "meat steak or related",
    "bird": "cat",
    "cow": "dog",
    "bus": "cow",
}

# Class-specific questions (overrides the default question for these classes)
QUESTIONS_PER_CLASS = {
    "cow":   "What is in the image and what animals can you see? Describe briefly.",
    "pizza": "What is in the image and what food can you see? Briefly describe. ",
    "meat steak or related": "What is in the image and what food can you see? Briefly describe. ",
}

# Generic fallback questions tried in order when the model emits invalid tokens
FALLBACK_QUESTIONS = [
    "What is in the image? Describe briefly",
    "What is in the image? Describe very briefly",
    "What is in the image?",
    "Describe the image content.",
    "Summarize the image in one sentence.",
]

INVALID_TOKENS = {
    "<end_of_turn>", "<end_of_sentence>", "<eos><end_of_turn>",
    "assistant<|eot_id|>", "assistantassistant", None,
}


def perform_steering(model, saes, image_processor, image_path,
                      steering_visual_token, add_feat, remove_feat,
                      language_model_key, add_coef=0, remove_coef=0,
                      steer_layer=(0, 1, 2),
                      question="What is shown in this image? Describe in 2 sentences.",
                      verbose=False, device="cuda"):
    """Run a single steered generation. Returns the decoded output text."""
    mcfg = config.get_model_config(language_model_key)
    is_gemma = "gemma" in language_model_key.lower()

    if is_gemma:
        prompt = (f"<start_of_turn>user\n{question}\n<image><end_of_turn>\n"
                   f"<start_of_turn>model\n")
    else:
        prompt = (f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
                   f"{question}\n<image><|eot_id|><|start_header_id|>model"
                   f"<|end_header_id|>\n\n")

    if verbose:
        print("Full question:", prompt)

    add_layer = add_feat["layer"]
    add_idx = add_feat["feat"]
    remove_layer = remove_feat["layer"]
    remove_idx = remove_feat["feat"]

    if verbose:
        _, desc_add = get_feature_information(add_layer, add_idx, model=mcfg["np_name"])
        _, desc_rem = get_feature_information(remove_layer, remove_idx, model=mcfg["np_name"])
        print(f"Add feat {add_idx} (layer {add_layer}): {desc_add}")
        print(f"Remove feat {remove_idx} (layer {remove_layer}): {desc_rem}")
        print(f"add_coef={add_coef}, remove_coef={remove_coef}")

    # SAE decoder rows for the picked features, unit-normalized
    add_vec = saes[add_layer].W_dec[add_idx, :].detach()
    remove_vec = saes[remove_layer].W_dec[remove_idx, :].detach()
    add_vec = add_vec / torch.norm(add_vec)
    remove_vec = remove_vec / torch.norm(remove_vec)

    tokenizer = model.tokenizer
    tokenized = tokenizer(prompt, return_tensors="pt", max_length=50, truncation=True)
    input_ids_ = tokenized["input_ids"].to(device)
    attention_mask_ = tokenized["attention_mask"].to(device)

    eos_token = mcfg["eos_token_id"]
    model_token = mcfg["model_token_id"]

    if is_gemma:
        cutoff = input_ids_[0].tolist().index(model_token) + 1
    else:
        # +2 because of the <|end_header_id|>\n\n tokens after "model"
        cutoff = input_ids_[0].tolist().index(model_token) + 2

    input_ids = input_ids_[:, :cutoff]
    attention_mask = attention_mask_[:, :cutoff]

    image = Image.open(image_path).convert("RGB")
    image_tensor = image_processor(images=image, return_tensors="pt")["pixel_values"].to(device)

    max_new_tokens = 100
    temperature = 0.0001  # near-greedy
    generated_tokens = []
    current_input_ids = input_ids.clone().to(device)
    current_attention_mask = attention_mask.clone().to(device)
    steering_visual_token = torch.tensor(steering_visual_token)

    for _ in range(max_new_tokens):
        with torch.no_grad():
            outputs = model(
                image=image_tensor,
                input_ids=current_input_ids,
                attention_mask=current_attention_mask,
                return_residuals=False,
                steer_layer=list(steer_layer),
                steer_visual_token_idx=steering_visual_token,
                add_vector=add_vec,
                add_coeff=add_coef,
                remove_vector=remove_vec,
                remove_coeff=remove_coef,
            )

        next_token_logits = outputs[:, -1, :] / temperature
        next_token_probs = torch.softmax(next_token_logits, dim=-1)
        next_token = next_token_probs.argmax(-1).squeeze()
        generated_tokens.append(next_token.item())

        if next_token.item() == eos_token:
            break

        current_input_ids = torch.cat(
            [current_input_ids, next_token.unsqueeze(0).unsqueeze(0)], dim=1,
        )
        current_attention_mask = torch.cat(
            [current_attention_mask, torch.ones((1, 1), device=device)], dim=1,
        )

    return model.language_model.to_string(generated_tokens)


def build_valid_images(coco_results, coco, vision_model_name, mapper, max_n=None,
                        image_filter=None):
    """For each single-annotation image, compute bbox -> visual-patch mapping.
 
    If `image_filter` is provided, only images whose path is in that set are kept;
    everything else is skipped before any expensive patch-grid plotting.
    """
    valid = []
    for img_key in tqdm(coco_results, desc="Mapping bboxes to patches"):
        data = coco_results[img_key]
        if image_filter is not None and data["image_path"] not in image_filter:
            continue
        if len(data["annotations"]) > 1:
            continue
        img = Image.open(data["image_path"])
        for ann in data["annotations"]:
            cat_info = coco.loadCats([ann["category_id"]])[0]
            bbox = transform_bbox_to_crop_space(ann["bbox"], img.size, vmodel=vision_model_name)
            patch_grid = mapper.get_patch_grid()
            mapping = mapper.bbox_to_patches(bbox, patch_grid)
            patch_indices = sorted(p["patch_idx"] for p in mapping["intersecting_patches"])
            img_pil = plot_model_patch_grid(
                img, model_type=vision_model_name,
                highlight_patches=patch_indices, include_grid=True,
            )
            valid.append((img_pil, data["image_path"], patch_indices, cat_info))
        if max_n is not None and len(valid) >= max_n:
            break
    return valid


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--language-model", default="meta-llama/Llama-3.1-8B-Instruct",
                    choices=list(LATENTS_BY_MODEL))
    p.add_argument("--vision-model", default="dinov2",
                    choices=["clip", "dinov2", "jepa"])
    p.add_argument("--projector-weights", required=True)
    p.add_argument("--coco-path", default=str(config.COCO_PATH))
    p.add_argument("--images-file", default=None,
                    help="Optional path to a Python file or JSON list of image paths "
                         "to restrict to (the original `steering_imgs.all_images`).")
    p.add_argument("--output", default=str(config.RESULTS_DIR / "steering_results.json"))
    p.add_argument("--max-results", type=int, default=5000)
    p.add_argument("--inject-coeff", type=float, default=10.0)
    p.add_argument("--remove-coeff", type=float, default=10.0)
    p.add_argument("--steer-layers", type=int, nargs="+", default=None,
                    help="Layer indices to apply steering at. Defaults to the model's "
                         "entry in MODEL_REGISTRY (Llama: [0,1], Gemma: [0,1,2]).")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def load_image_filter(images_file):
    if images_file is None:
        return None
    p = Path(images_file)
    if p.suffix == ".json":
        return set(json.loads(p.read_text()))
    # Python module path: import all_images from it
    import importlib.util
    spec = importlib.util.spec_from_file_location("_steering_imgs", str(p))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return set(mod.all_images)


def main():
    args = parse_args()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    latents = LATENTS_BY_MODEL[args.language_model]
    mcfg = config.get_model_config(args.language_model)
    steer_layers = args.steer_layers if args.steer_layers is not None else mcfg["steer_layers"]
    print(f"Steering layers: {steer_layers}")
    
    image_filter = load_image_filter(args.images_file)
    print("image_filter", image_filter)
    
    # print("SAVING", args.output)
    # Path(args.output).write_text(json.dumps(image_filter, indent=2, default=str))

    # --- Build models ---
    language_model = load_language_model(args.language_model, use_processing=False,
                                          device=args.device)
    vision_model, image_processor = get_vision_model(args.vision_model, device=args.device)
    saes = {i: sae for i, sae in enumerate(load_saes(args.language_model, device=args.device))}

    model = LLaVAModel(vision_model, language_model, language_model.tokenizer).to(args.device)
    model.load_projector_weights(args.projector_weights)

    
    # --- Build valid (image, bbox, patches, class) tuples from COCO ---
    coco_results, coco = extract_coco_classes(
        coco_path=args.coco_path,
        classes=['dog', 'cat', 'cow', 'bird', 'bus', 'pizza'],
        output_dir="./extracted_coco",
        max_results=args.max_results,
        splits=["val"],
    )
    mapper = VisionPatchMapper(vision_model, image_processor)
    valid_images = build_valid_images(
        coco_results, coco, args.vision_model, mapper,
        image_filter=image_filter,
    )
    print(f"{len(valid_images)} valid (image, single-bbox) pairs")

    
    # --- Main loop ---
    all_results = {}
    counter = 0
    for img_w_patch, img_path, steer_patches, class_info in tqdm(valid_images, desc="Steering"):
        if image_filter is not None and img_path not in image_filter:
            continue

        remove_concept = class_info["name"]
        if remove_concept not in CLASS_MAP:
            continue
        inject_concept = CLASS_MAP[remove_concept]

        remove_latent = latents[remove_concept]
        add_latent = latents[inject_concept]

        baseline = removal = inject = None
        success = False

        for current_q in FALLBACK_QUESTIONS:
            q_main = current_q
            q_inject = (QUESTIONS_PER_CLASS.get(inject_concept, current_q)
                         if current_q == FALLBACK_QUESTIONS[0] else current_q)

            if baseline in INVALID_TOKENS:
                baseline = perform_steering(
                    model, saes, image_processor, img_path, steer_patches,
                    add_feat=add_latent, remove_feat=remove_latent,
                    language_model_key=args.language_model,
                    add_coef=0, remove_coef=0, question=q_main, device=args.device,
                )
            if removal in INVALID_TOKENS:
                removal = perform_steering(
                    model, saes, image_processor, img_path, steer_patches,
                    add_feat=add_latent, remove_feat=remove_latent,
                    language_model_key=args.language_model,
                    add_coef=0, remove_coef=args.remove_coeff,
                    steer_layer=steer_layers, question=q_main, device=args.device,
                )
            if inject in INVALID_TOKENS:
                inject = perform_steering(
                    model, saes, image_processor, img_path, steer_patches,
                    add_feat=add_latent, remove_feat=remove_latent,
                    language_model_key=args.language_model,
                    add_coef=args.inject_coeff, remove_coef=0,
                    steer_layer=steer_layers, question=q_inject, device=args.device,
                )
            if all(r not in INVALID_TOKENS for r in (baseline, removal, inject)):
                success = True
                break
            print(f"Partial failure for {img_path}; retrying with: {current_q!r}")

        if not success:
            failed = {"target_status": "Fail", "score": 0}
            all_results[img_path] = {
                "remove_concept": remove_concept,
                "inject_concept": inject_concept,
                "baseline": baseline, "removal": removal, "inject": inject,
                "resp_remove": failed, "resp_inject": failed,
            }
            continue

        print(f"Remove {remove_concept!r}, Inject {inject_concept!r}")
        print(f"Baseline: {baseline}\nRemoval:  {removal}\nInject:   {inject}")

        image = Image.open(img_path)
        resp_remove = evaluate_interpretability(
            image, keyword=remove_concept, steered_text=removal,
            baseline_text=baseline, concept_type="remove",
        )
        resp_inject = evaluate_interpretability(
            image, keyword=inject_concept, steered_text=inject,
            source_concept=remove_concept, baseline_text=baseline,
            concept_type="inject",
        )

        all_results[img_path] = {
            "remove_concept": remove_concept,
            "inject_concept": inject_concept,
            "inject_coeff": args.inject_coeff,
            "remove_coeff": args.remove_coeff,
            "baseline": baseline, "removal": removal, "inject": inject,
            "resp_remove": resp_remove, "resp_inject": resp_inject,
        }
        counter += 1
        print(f"counter: {counter}")

        # Save incrementally so we don't lose work on a crash
        Path(args.output).write_text(json.dumps(all_results, indent=2, default=str))

    print(f"\nFinal results saved to {args.output}")


if __name__ == "__main__":
    main()
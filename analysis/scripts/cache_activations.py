import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import gc
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from vlm_sae import config
from vlm_sae.dataset_llava import CacheDataset
from vlm_sae.models import load_language_model, load_saes
from vlm_sae.vision import get_vision_model


def get_image_hook(image_features):
    """Replace the first 256 residual-stream positions with the projected image features."""
    def image_hook(act, hook):
        if act.shape[1] > 1:
            act[:, 1:257] = image_features
        return act
    return image_hook


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cache_n_examples", type=int, default=50)
    p.add_argument("--output_dir", type=str, default=str(config.ACTIVATIONS_DIR))
    p.add_argument("--projector_weights", type=str, required=True)
    p.add_argument("--json_file", type=str, default=str(config.LLAVA_JSON))
    p.add_argument("--image_dir", type=str, default=str(config.LLAVA_IMAGE_DIR))
    p.add_argument("--vision_model", type=str, default="clip",
                    choices=["clip", "dinov2", "jepa"])
    p.add_argument("--use_images", type=str, default="")
    p.add_argument("--language_model", type=str, default="gemma-2-2b-it")
    return p.parse_args()


def main():
    args = parse_args()
    print("Args", args)
    os.makedirs(args.output_dir, exist_ok=True)

    mcfg = config.get_model_config(args.language_model)
    num_layers = mcfg["num_layers"]
    split_token = mcfg["split_token"]
    model_start_token = mcfg["model_start"]
    is_llama = "llama" in args.language_model.lower()

    print(f"Language model: {args.language_model} ({num_layers} layers)")

    saes = load_saes(args.language_model, device="cuda")
    model = load_language_model(args.language_model, use_processing=True, device="cuda")

    print(f"Loading vision model: {args.vision_model}")
    vision_model, image_processor = get_vision_model(args.vision_model, device="cuda")
    vcfg = config.get_vision_config(args.vision_model)
    img_dim = vcfg["img_dim"]

    # Projector
    projector_state_dict = torch.load(args.projector_weights)
    projector = nn.Linear(vision_model.config.hidden_size, model.cfg.d_model).to("cuda")
    projector.load_state_dict(projector_state_dict)

    dataset = CacheDataset(
        json_file=args.json_file,
        image_dir=args.image_dir,
        tokenizer=model.tokenizer,
        image_processor=image_processor,
        model_type="llama" if is_llama else "gemma",
    )


    if args.use_images:
        with open(args.use_images, "r") as f:
            use_images = f.read()
        args.cache_n_examples = 10 ** 8
        seen_images = []
    else:
        use_images = []

    example_id = 0
    n = min(len(dataset), args.cache_n_examples)
    for dataset_id in tqdm(range(n), desc="Processing examples"):
        # print("IMAGE PATH", dataset[dataset_id]["image_path"])
        if use_images:
            if (dataset[dataset_id]["image_path"] not in use_images
                    or dataset[dataset_id]["image_path"] in seen_images):
                continue
            seen_images.append(dataset[dataset_id]["image_path"])

        input_ids = dataset[dataset_id]["input_ids"].to("cuda")
        assistant_positions = dataset[dataset_id]["assistant_positions"][0]
        answer_clue = model.tokenizer.decode(
            input_ids[assistant_positions[0]:assistant_positions[1] - 1]
        )
        image_tensor = dataset[dataset_id]["image_tensor"]

        with torch.no_grad():
            image = vision_model(image_tensor.to("cuda"))
            image_features = projector(image.last_hidden_state)
            if args.vision_model == "jepa":
                image_features = image_features.detach()  # no CLS token in JEPA
            else:
                image_features = image_features.detach()[:, 1:]

            decoded_text = model.tokenizer.decode(input_ids[1:])  # skip BOS
            user_part = decoded_text.split(split_token)[0]
            prompt = user_part + model_start_token
            text_prompt = prompt.replace(
                "<image>", f"Consider the following information: {answer_clue} ",
            )

            input_ids = model.tokenizer.encode(prompt, return_tensors="pt").to("cuda")[0]
            # Insert placeholders for the visual tokens after BOS
            input_ids = torch.cat([
                input_ids[:1],
                torch.zeros(img_dim, dtype=torch.int64).to("cuda"),
                input_ids[1:],
            ], dim=0)

            text_input_ids = model.tokenizer.encode(text_prompt, return_tensors="pt").to("cuda")[0]

            with model.hooks(fwd_hooks=[("blocks.0.hook_resid_pre", get_image_hook(image_features))]):
                _, cache = model.run_with_cache(input_ids.unsqueeze(0))
            _, text_cache = model.run_with_cache(text_input_ids.unsqueeze(0))

        projector_outputs = cache["blocks.0.hook_resid_pre"][:, 1:257, :]
        example_data = {"projector_outputs": projector_outputs.cpu()}

        for layer_idx, sae in enumerate(saes):
            layer_image_acts = cache[f"blocks.{layer_idx}.hook_resid_post"][:, 1:257, :]
            layer_image_text_acts = cache[f"blocks.{layer_idx}.hook_resid_post"][:, 257:, :]
            layer_text_acts = text_cache[f"blocks.{layer_idx}.hook_resid_post"][:, 1:, :]


            sae_image_acts = sae.encode(layer_image_acts)
            sae_image_text_acts = sae.encode(layer_image_text_acts)
            sae_text_acts = sae.encode(layer_text_acts)

            sae_image_recon = sae.decode(sae_image_acts)
            sae_image_text_recon = sae.decode(sae_image_text_acts)
            sae_text_recon = sae.decode(sae_text_acts)

            example_data[f"layer_{layer_idx}"] = {
                "image_activations": layer_image_acts.cpu(),
                "image_text_activations": layer_image_text_acts.cpu(),
                "text_activations": layer_text_acts.cpu(),
                "sae_image_activations": sae_image_acts.cpu(),
                "sae_image_text_activations": sae_image_text_acts.cpu(),
                "sae_text_activations": sae_text_acts.cpu(),
                "sae_image_reconstructions": sae_image_recon.cpu(),
                "sae_image_text_reconstructions": sae_image_text_recon.cpu(),
                "sae_text_reconstructions": sae_text_recon.cpu(),
            }

        save_path = os.path.join(args.output_dir, f"example_{dataset_id}.pt")
        torch.save(example_data, save_path)
        example_id += 1

        del cache, example_data, text_cache
        torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    main()


import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import gc
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from vlm_sae import config
from vlm_sae.api import evaluate_sae_features_batch, get_feature_information
from vlm_sae.dataset_llava import GemmaLLaVADatasetSimplified
from vlm_sae.data import load_activations


def resolve_indices(start, end, scan_dir, activations_dir):
    """Return [start..end) indices, either as ints or as filename-derived keys.

    When `scan_dir` is True we list the activations directory and take the
    `start:end` slice of whatever's there — needed when caching was sparse
    or filenames aren't 0..N-1.
    """
    if not scan_dir:
        return list(range(start, end))
    names = sorted(os.listdir(activations_dir))
    sliced = names[start:end]
    return [n.replace("example_", "").replace(".pt", "") for n in sliced]


def compute_feature_frequencies(num_freq_features, token_types, num_layers, batch_size,
                                  cache_path=None, recompute=False, base_dir=None,
                                  scan_dir=False):
    """Average per-feature activation rate across `num_freq_features` cached samples."""
    if cache_path and Path(cache_path).exists() and not recompute:
        print(f"Loading cached frequencies from {cache_path}")
        return torch.load(cache_path)

    feature_frequencies = {m: [None] * num_layers for m in token_types}
    frequency_counts = {m: [0] * num_layers for m in token_types}

    for m in token_types:
        for start_idx in tqdm(range(0, num_freq_features, batch_size),
                               desc=f"Frequencies ({m})"):
            end_idx = min(start_idx + batch_size, num_freq_features)
            batch_indices = resolve_indices(start_idx, end_idx, scan_dir, base_dir)

            sae_activations = load_activations(
                computation="sae_encoded", layers=list(range(num_layers)),
                indices=batch_indices, model_type=m, base_dir=base_dir,
            )
            for layer_idx in range(num_layers):
                stacked = torch.cat(
                    [x.mean(dim=1) for x in sae_activations[layer_idx]], dim=0,
                )
                binary = (stacked > 0).float().mean(dim=0, keepdim=True)
                if feature_frequencies[m][layer_idx] is None:
                    feature_frequencies[m][layer_idx] = binary
                else:
                    feature_frequencies[m][layer_idx] += binary
                frequency_counts[m][layer_idx] += 1

            del sae_activations
            torch.cuda.empty_cache()
            gc.collect()

    for m in token_types:
        for layer_idx in range(num_layers):
            feature_frequencies[m][layer_idx] /= frequency_counts[m][layer_idx]

    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(feature_frequencies, cache_path)
    return feature_frequencies


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--language-model", default="gemma-2-2b-it")
    p.add_argument("--n_features_per_layer", type=int, default=3)
    p.add_argument("--frequency_threshold", type=float, default=0.05,
                    help="Mask out features that fire above this rate on average.")
    p.add_argument("--density_threshold", type=float, default=0.005,
                    help="Drop Neuronpedia features denser than this.")
    p.add_argument("--batch_size", type=int, default=10)
    p.add_argument("--num_samples", type=int, default=100,
                    help="Number of examples to evaluate.")
    p.add_argument("--num_freq_features", type=int, default=100,
                    help="Number of examples to compute frequency stats from.")
    p.add_argument("--token_types", nargs="+",
                    default=["vlm_img"],
                    choices=["llm_text", "vlm_img", "vlm_imgtext"])
    p.add_argument("--activations_dir", default=str(config.ACTIVATIONS_DIR))
    p.add_argument("--json_file", default=str(config.LLAVA_JSON))
    p.add_argument("--image_dir", default=str(config.LLAVA_IMAGE_DIR))
    p.add_argument("--output", default=str(config.RESULTS_DIR / "matching_rate_results.json"))
    p.add_argument("--plot", default=str(config.RESULTS_DIR / "matching_rate.pdf"))
    p.add_argument("--freq_cache", default=None,
                    help="Where to cache feature frequencies. Auto-named if unset.")
    p.add_argument("--auditor-model", default="gemini-2.5-flash-lite",
                    help="Gemini model used by evaluate_sae_features_batch.")
    p.add_argument("--scan-dir", action="store_true",
                    help="Discover example indices by listing the activations directory "
                         "instead of assuming dense 0..N-1. Use this when caching was "
                         "sparse (e.g. ran with --use_images) or filename-keyed.")
    p.add_argument("--no-plot", action="store_true",
                help="Skip writing the per-layer match rate plot.")
    return p.parse_args()


def main():
    args = parse_args()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    mcfg = config.get_model_config(args.language_model)
    num_layers = mcfg["num_layers"]
    np_name = mcfg["np_name"]

    dataset = GemmaLLaVADatasetSimplified(
        json_file=args.json_file, image_dir=args.image_dir,
        tokenizer=None, image_processor=None,
    )

    freq_cache = args.freq_cache or str(
        config.RESULTS_DIR / "vars" /
        f"feature_frequencies_nfreq{args.num_freq_features}"
        f"_nsamples{args.num_samples}"
        f"_{args.language_model.replace('/', '_')}.pt"
    )
    feature_frequencies = compute_feature_frequencies(
        num_freq_features=args.num_freq_features,
        token_types=args.token_types,
        num_layers=num_layers,
        batch_size=args.batch_size,
        cache_path=freq_cache,
        base_dir=args.activations_dir,
        scan_dir=args.scan_dir,
    )

    results = {"layer": [], "token_type": [], "match_rate": []}
    detailed_results = {"evaluations": []}
    tested_ids = set()

    for token_type in args.token_types:
        print(f"Evaluating {token_type} features")
        progress = tqdm(total=args.num_samples, desc=f"Token type: {token_type}")

        for batch_start in range(0, args.num_samples, args.batch_size):
            batch_end = min(batch_start + args.batch_size, args.num_samples)
            batch_indices = resolve_indices(
                batch_start, batch_end, args.scan_dir, args.activations_dir,
            )

            sae_activations = load_activations(
                computation="sae_encoded", layers=list(range(num_layers)),
                indices=batch_indices, model_type=token_type,
                base_dir=args.activations_dir,
            )

            for batch_idx, sample_idx in enumerate(batch_indices):
                # When indices come from filenames they're strings; cast to int
                # for the dataset lookup if the dataset is integer-keyed.
                try:
                    sample = dataset.__getitem__(int(sample_idx))
                except (ValueError, TypeError):
                    sample = dataset.__getitem__(sample_idx)
                unique_id = sample["unique_id"]
                image_id = sample["image_id"]
                image = sample["image"]

                if (token_type, unique_id) in tested_ids:
                    progress.update(1)
                    continue

                for layer_idx in range(num_layers):
                    activations = sae_activations[layer_idx][batch_idx][0].to("cuda")
                    freq_mask = (feature_frequencies[token_type][layer_idx].to("cuda")
                                  <= args.frequency_threshold)
                    activations = activations * freq_mask

                    flat = activations.float().flatten().detach()
                    top_indices = flat.argsort(descending=True)

                    descs, feat_idxs, tok_idxs, act_vals = [], [], [], []
                    top_idx = 0
                    while len(descs) < args.n_features_per_layer:
                        if top_idx >= top_indices.numel():
                            break
                        feat_flat = top_indices[top_idx]
                        feat = feat_flat % activations.shape[1]
                        tok = feat_flat // activations.shape[1]
                        top_idx += 1

                        density, desc = get_feature_information(
                            layer=layer_idx, feature=feat.item(), model=np_name,
                        )
                        if density is None or desc is None:
                            continue
                        if density > args.density_threshold:
                            continue
                        if feat.item() in feat_idxs:
                            continue

                        if flat[feat_flat].item() <= 0:
                            # Below zero -> empty description so it can't false-match
                            descs.append("")
                        else:
                            descs.append(desc)
                        feat_idxs.append(feat.item())
                        tok_idxs.append(tok.item())
                        act_vals.append(flat[feat_flat].item())

                    if not descs:
                        continue

                    match_result, explanation, feature_ids_match, full_expl = 0, "", [], {}
                    for att in range(5):
                        try:
                            match_result, explanation, feature_ids_match, full_expl = (
                                evaluate_sae_features_batch(descs, image,
                                                              model=args.auditor_model)
                            )
                            break
                        except Exception as e:
                            print(f"Attempt {att} failed: {e}")
                            time.sleep(1)

                    detailed_results["evaluations"].append({
                        "layer": layer_idx,
                        "token_type": token_type,
                        "token_idxs": tok_idxs,
                        "act_values": act_vals,
                        "feature_ids_match": feature_ids_match,
                        "feature_indices": feat_idxs,
                        "feature_descriptions": descs,
                        "match_found": match_result,
                        "explanation": full_expl,
                        "unique_id": unique_id,
                        "image_id": image_id,
                    })
                    results["layer"].append(layer_idx)
                    results["token_type"].append(token_type)
                    results["match_rate"].append(match_result)

                tested_ids.add((token_type, unique_id))
                progress.update(1)

            del sae_activations
            torch.cuda.empty_cache()
            gc.collect()

    Path(args.output).write_text(json.dumps(detailed_results, indent=2, default=str))
    print(f"Detailed results saved to {args.output}")
    
    if args.no_plot:
        return

    # --- Plot ---
    df = pd.DataFrame(results)
    df_avg = df.groupby(["layer", "token_type"])["match_rate"].mean().reset_index()
    plt.figure(figsize=(12, 5))
    layers = sorted(df_avg["layer"].unique())
    x = np.arange(len(layers))
    for i, tt in enumerate(args.token_types):
        token_data = df_avg[df_avg["token_type"] == tt].sort_values("layer")
        plt.scatter(x, token_data["match_rate"], label=tt, alpha=0.8)
        plt.plot(x, token_data["match_rate"], alpha=0.8)
    plt.xlabel("Layer")
    plt.ylabel("Match Rate")
    plt.title("SAE Feature / Image Concept Match Rate Across Layers")
    plt.xticks(x, layers)
    plt.legend()
    plt.grid(True, axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    Path(args.plot).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.plot, dpi=200, bbox_inches="tight")
    print(f"Plot saved to {args.plot}")


if __name__ == "__main__":
    main()
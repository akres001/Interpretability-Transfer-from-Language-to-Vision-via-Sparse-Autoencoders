import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import gc
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from tqdm import tqdm

from vlm_sae import config
from vlm_sae.data import load_activations_batch


# Display name mapping used in the legend
TOKEN_TYPE_LABEL = {
    "vlm_img":     "VLM Image Positions",
    "vlm_imgtext": "VLM Text Positions",
    "llm_text":    "LLM Prompt Positions",
}

# Colorblind-friendly palette (matches the original notebook's choices)
PALETTE = {
    "LLM Prompt Positions":  "#1f77b4",
    "VLM Image Positions":   "#ff7f0e",
    "VLM Text Positions":    "#2ca02c",
}


def _create_subplot(df, y_column, title, y_label, palette, y_lim=''):
    """
    Helper function to create a subplot with the given data.
    
    Args:
        df: DataFrame containing the data
        y_column: Column name for the y-axis values
        title: Title for the plot
        y_label: Label for the y-axis
        palette: Color palette to use (dictionary mapping token types to colors)
    """
    # Create the line plot
    ax = sns.lineplot(
        data=df,
        x='layer',
        y=y_column,
        hue='token_type',
        palette=palette,
        linewidth=2.5,
        marker='o',
        markersize=8,
        markeredgecolor='white',
        markeredgewidth=1.5
    )
    
    # Set y-axis limits with some padding
    min_y = df[y_column].min()
    max_y = df[y_column].max()
    y_padding = (max_y - min_y) * 0.05  # 5% padding
    if y_lim:
        plt.ylim(y_lim[0], y_lim[1])
    else:
        plt.ylim(max(0, min_y - y_padding), max_y + y_padding)
    
    # Customize the plot for publication quality with increased font sizes
    plt.xlabel('Layer', fontsize=18, fontweight='bold')
    plt.ylabel(y_label, fontsize=18, fontweight='bold')
    plt.title(title, fontsize=18, fontweight='bold', pad=20)
    
    # Move legend inside the figure
    plt.legend(title='Token Type', title_fontsize=18, fontsize=18, 
               frameon=True, facecolor='white', edgecolor='lightgray',
               loc='best')
    
    # Add grid for better readability
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Ensure x-axis shows only every second layer value with larger font
    layers = sorted(df['layer'].unique())
    # Show only every second layer tick
    layers_to_show = layers[::2]
    # Create empty labels for the layers we want to hide
    all_labels = []
    for layer in layers:
        if layer in layers_to_show:
            all_labels.append(str(layer))
        else:
            all_labels.append('')
    
    plt.xticks(layers, all_labels, fontsize=18)
    plt.yticks(fontsize=18)
    
    # Add subtle shading under the lines
    for token_type in df['token_type'].unique():
        token_data = df[df['token_type'] == token_type].sort_values('layer')
        plt.fill_between(
            token_data['layer'], 
            token_data[y_column], 
            alpha=0.1, 
            color=palette[token_type]
        )
    
    return ax


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--language-model", default="gemma-2-2b-it")
    p.add_argument("--num_samples", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=10)
    p.add_argument("--token_types", nargs="+",
                    default=["llm_text", "vlm_img", "vlm_imgtext"],
                    choices=["llm_text", "vlm_img", "vlm_imgtext"])
    p.add_argument("--activations_dir", default=str(config.ACTIVATIONS_DIR))
    p.add_argument("--plot", default=str(config.RESULTS_DIR / "reconstruction_sparsity.pdf"))
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    mcfg = config.get_model_config(args.language_model)
    num_layers = mcfg["num_layers"]

    # Per-token-type, per-layer running sums + counts
    metrics_sum = {tt: {"l2": [0.0] * num_layers, "l0": [0.0] * num_layers}
                    for tt in args.token_types}
    sample_counts = {tt: [0] * num_layers for tt in args.token_types}

    for token_type in args.token_types:
        print(f"Processing {token_type}...")
        for start_idx in tqdm(range(0, args.num_samples, args.batch_size),
                               desc=f"Batches ({token_type})"):
            end_idx = min(start_idx + args.batch_size, args.num_samples)

            original_acts = load_activations_batch(
                computation=None, layers=list(range(num_layers)),
                start_idx=start_idx, end_idx=end_idx,
                model_type=token_type, base_dir=args.activations_dir,
            )
            reconstructed_acts = load_activations_batch(
                computation="sae_decoded", layers=list(range(num_layers)),
                start_idx=start_idx, end_idx=end_idx,
                model_type=token_type, base_dir=args.activations_dir,
            )
            features = load_activations_batch(
                computation="sae_encoded", layers=list(range(num_layers)),
                start_idx=start_idx, end_idx=end_idx,
                model_type=token_type, base_dir=args.activations_dir,
            )

            for layer_idx in range(num_layers):
                if layer_idx not in original_acts:
                    print(f"  Warning: layer {layer_idx} missing for batch "
                           f"{start_idx}-{end_idx - 1}")
                    continue
                for sample_idx in range(len(original_acts[layer_idx])):
                    orig = original_acts[layer_idx][sample_idx][0].to(args.device)
                    recon = reconstructed_acts[layer_idx][sample_idx][0].to(args.device)
                    feat = features[layer_idx][sample_idx][0].to(args.device)

                    l2 = (recon - orig).pow(2).mean().item()
                    l0 = (feat != 0).float().mean().item()

                    metrics_sum[token_type]["l2"][layer_idx] += l2
                    metrics_sum[token_type]["l0"][layer_idx] += l0
                    sample_counts[token_type][layer_idx] += 1

                    del orig, recon, feat

            del original_acts, reconstructed_acts, features
            torch.cuda.empty_cache()
            gc.collect()

    # --- Reshape into long DataFrames for plotting ---
    rows_l2, rows_l0 = [], []
    for tt in args.token_types:
        for layer_idx in range(num_layers):
            c = sample_counts[tt][layer_idx]
            l2_avg = metrics_sum[tt]["l2"][layer_idx] / c if c else 0
            l0_avg = metrics_sum[tt]["l0"][layer_idx] / c if c else 0
            rows_l2.append({"layer": layer_idx, "token_type": TOKEN_TYPE_LABEL[tt],
                             "value": l2_avg})
            rows_l0.append({"layer": layer_idx, "token_type": TOKEN_TYPE_LABEL[tt],
                             "value": l0_avg})

    df_l2 = pd.DataFrame(rows_l2)
    df_l0 = pd.DataFrame(rows_l0)

    # --- Plot ---
    plt.figure(figsize=(20, 6))
    sns.set_style("whitegrid")
    plt.subplot(1, 2, 1)
    _create_subplot(df_l2, "value", "SAE Reconstruction Loss Across Layers",
                     "Reconstruction Loss (MSE)", PALETTE)
    plt.subplot(1, 2, 2)
    _create_subplot(df_l0, "value", "SAE Active Feature Fraction Across Layers",
                     "L0 Density", PALETTE)
    plt.tight_layout()
    Path(args.plot).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.plot, dpi=1200, format="pdf", bbox_inches="tight",
                 pad_inches=0.05, facecolor="white")
    print(f"Plot saved to {args.plot}")


if __name__ == "__main__":
    main()
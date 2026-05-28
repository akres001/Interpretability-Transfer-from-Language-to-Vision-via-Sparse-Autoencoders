import os

import torch
import torch.nn as nn
from sae_lens import SAE
from tqdm import tqdm
from transformer_lens import HookedTransformer

from . import config


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def load_language_model(model_key: str, dtype="bfloat16",
                         use_processing: bool = False, device: str = "cuda"):
    """Wrap HookedTransformer.from_pretrained{_no_processing}.

    `use_processing=False` matches the no-processing variant used by the
    steering notebook; activation caching used the processing variant.
    """
    if use_processing:
        return HookedTransformer.from_pretrained(model_key).to(device)
    return HookedTransformer.from_pretrained_no_processing(model_key, dtype=dtype).to(device)


def load_saes(language_model: str, device: str = "cuda", num_layers: int = None):
    """Load all SAEs for the language model, in layer order."""
    cfg = config.get_model_config(language_model)
    n = num_layers if num_layers is not None else cfg["num_layers"]
    saes = []
    for layer in tqdm(range(n), desc="Loading SAEs"):
        sae, _, _ = SAE.from_pretrained(
            release=cfg["sae_release"],
            sae_id=cfg["sae_id_fmt"].format(layer=layer),
            device=device,
        )
        saes.append(sae)
    return saes


# ---------------------------------------------------------------------------
# LLaVA model (for steering)
# ---------------------------------------------------------------------------
class LLaVAModel(nn.Module):
    """LLaVA-style wrapper: frozen vision + frozen LM + trainable linear projector.

    Supports SAE-direction-based steering on visual tokens via the
    add_vector / remove_vector arguments to forward().
    """

    def __init__(self, vision_model, language_model, tokenizer, projection_dim=None):
        super().__init__()
        self.projection_dim = projection_dim or language_model.cfg.d_model
        self.vision_model = vision_model
        self.language_model = language_model
        self.tokenizer = tokenizer
        self.projection = self.build_vision_projector(
            vision_model.config.hidden_size, self.projection_dim,
        )
        for param in self.vision_model.parameters():
            param.requires_grad = False
        for param in self.language_model.parameters():
            param.requires_grad = False

    def build_vision_projector(self, input_dim, output_dim):
        return nn.Linear(input_dim, output_dim)

    def load_projector_weights(self, path):
        path_ = path + ".pth" if not str(path).endswith(".pth") else str(path)
        if os.path.exists(path_):
            d = torch.load(path_)
            # Handle DDP-wrapped state dicts
            if "module.weight" in d:
                d["weight"] = d.pop("module.weight")
                d["bias"] = d.pop("module.bias")
            self.projection.load_state_dict(d)
            print(f"Loaded projector weights from {path_}")
        else:
            print(f"No projector weights found at {path_}")

    def save_projector_weights(self, path):
        torch.save(self.projection.state_dict(), path)

    def forward(
        self,
        image,
        input_ids,
        attention_mask,
        return_residuals=False,
        # SAE steering arguments
        steer_layer=None,
        steer_visual_token_idx=0,
        add_vector=None,
        add_coeff=1.0,
        remove_vector=None,
        remove_coeff=1.0,
    ):
        vision_outputs = self.vision_model(image)
        # Drop CLS token before projection (drop_cls=True for clip/dinov2)
        image_features = self.projection(vision_outputs.last_hidden_state[:, 1:])
        num_visual_tokens = image_features.shape[1]

        input_embeds = self.language_model.embed(input_ids)
        insert_position = 1  # after BOS

        prefix_embeds = input_embeds[:, :insert_position]
        suffix_embeds = input_embeds[:, insert_position:]
        combined_embeds = torch.cat([prefix_embeds, image_features, suffix_embeds], dim=1)

        prefix_attention = attention_mask[:, :insert_position]
        suffix_attention = attention_mask[:, insert_position:]
        vision_attention = torch.ones(
            attention_mask.shape[0], num_visual_tokens, device=attention_mask.device,
        )
        combined_attention_mask = torch.cat(
            [prefix_attention, vision_attention, suffix_attention], dim=1,
        ).to(torch.int64)

        x = combined_embeds

        residuals = {} if return_residuals else None
        steer_layers = ([int(l) for l in steer_layer] if steer_layer is not None else [])

        for i, block in enumerate(self.language_model.blocks):
            x = block(x, attention_mask=combined_attention_mask)

            if i in steer_layers:
                abs_token_idx = insert_position + steer_visual_token_idx

                if add_vector is not None:
                    v = add_vector.to(x.device, dtype=x.dtype)
                    x[:, abs_token_idx, :] = x[:, abs_token_idx, :] + (v * add_coeff)

                if remove_vector is not None:
                    v = remove_vector.to(x.device, dtype=x.dtype)
                    x[:, abs_token_idx, :] = x[:, abs_token_idx, :] - (v * remove_coeff)

            if return_residuals:
                residuals[i] = x

        x = self.language_model.ln_final(x)
        logits = x @ self.language_model.W_U

        if return_residuals:
            return logits, residuals
        return logits
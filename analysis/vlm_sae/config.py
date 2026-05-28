import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


# --- API keys ---
HF_TOKEN = os.environ.get("HF_TOKEN", "")
NEURONPEDIA_API_KEY = os.environ.get("NEURONPEDIA_API_KEY", "")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
GOOGLE_APPLICATION_CREDENTIALS = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")
VERTEX_PROJECT = os.environ.get("VERTEX_PROJECT", "testapi-491315")
VERTEX_LOCATION = os.environ.get("VERTEX_LOCATION", "global")

# Push HF token + neuronpedia key into env for SDKs that read them on import
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN
if NEURONPEDIA_API_KEY:
    os.environ["NEURONPEDIA_API_KEY"] = NEURONPEDIA_API_KEY
if GOOGLE_APPLICATION_CREDENTIALS:
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GOOGLE_APPLICATION_CREDENTIALS


# --- Paths ---
PROJECT_ROOT = Path(os.environ.get("VLM_SAE_ROOT", Path(__file__).resolve().parents[2]))
DATA_DIR = Path(os.environ.get("VLM_SAE_DATA", PROJECT_ROOT / "data"))
COCO_PATH = Path(os.environ.get("COCO_PATH", DATA_DIR / "coco"))
LLAVA_JSON = Path(os.environ.get(
    "LLAVA_JSON", DATA_DIR / "LLaVA-Instruct" / "llava_v1_5_mix665k.json"))
LLAVA_IMAGE_DIR = Path(os.environ.get("LLAVA_IMAGE_DIR", DATA_DIR / "LLaVA-Instruct"))
ACTIVATIONS_DIR = Path(os.environ.get("ACTIVATIONS_DIR", PROJECT_ROOT / "activations"))
WEIGHTS_DIR = Path(os.environ.get("WEIGHTS_DIR", PROJECT_ROOT / "train" / "weights"))
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", PROJECT_ROOT / "results"))


# --- Model registry ---
# Replaces every IS_GEMMA / IS_LLAMA branch and the if/elif for NUM_LAYERS.
# np_name and np_sae_fmt are the names used by neuronpedia for description lookup.
# split_token / model_start match the chat template used during caching.
MODEL_REGISTRY = {
    "gemma-2-2b-it": {
        "num_layers": 26,
        "sae_release": "gemma-scope-2b-pt-res-canonical",
        "sae_id_fmt": "layer_{layer}/width_16k/canonical",
        "np_name": "gemma-2-2b",
        "np_sae_fmt": "{layer}-gemmascope-res-16k",
        "split_token": "<start_of_turn>model",
        "model_start": "<start_of_turn>model\n",
        "eos_token_id": 107,
        "model_token_id": 2516,
        "steer_layers": [0, 1, 2],
    },
    "gemma-2-9b-it": {
        "num_layers": 42,
        "sae_release": "gemma-scope-9b-pt-res-canonical",
        "sae_id_fmt": "layer_{layer}/width_16k/canonical",
        "np_name": "gemma-2-9b",
        "np_sae_fmt": "{layer}-gemmascope-res-16k",
        "split_token": "<start_of_turn>model",
        "model_start": "<start_of_turn>model\n",
        "eos_token_id": 107,
        "model_token_id": 2516,
        "steer_layers": [0, 1, 2],
    },
    "meta-llama/Llama-3.1-8B-Instruct": {
        "num_layers": 32,
        "sae_release": "llama_scope_lxr_8x",
        "sae_id_fmt": "l{layer}r_8x",
        "np_name": "llama3.1-8b",
        "np_sae_fmt": "{layer}-llamascope-res-32k",
        "split_token": "<|start_header_id|>model<|end_header_id|>",
        "model_start": "<|start_header_id|>model<|end_header_id|>\n\n",
        "eos_token_id": 128009,
        "model_token_id": 2590,
        "steer_layers": [0, 1],
    },
}


# --- Vision encoder registry ---
# img_dim is the number of visual tokens after dropping CLS (if any).
# drop_cls=True means we slice off the [CLS] token before projection.
VISION_REGISTRY = {
    "clip": {
        "hf": "openai/clip-vit-large-patch14",
        "img_dim": 256,
        "drop_cls": True,
        "shortest_edge": 224,
    },
    "dinov2": {
        "hf": "facebook/dinov2-large",
        "img_dim": 256,
        "drop_cls": True,
        "shortest_edge": 256,
    },
    "jepa": {
        "hf": "facebook/ijepa_vith14_1k",
        "img_dim": 256,
        "drop_cls": False,
        "shortest_edge": 224,
    },
}


def get_model_config(language_model: str) -> dict:
    """Look up the entry in MODEL_REGISTRY, raising a clear error if missing."""
    if language_model not in MODEL_REGISTRY:
        raise KeyError(
            f"Unknown language model {language_model!r}. "
            f"Known: {list(MODEL_REGISTRY)}"
        )
    return MODEL_REGISTRY[language_model]


def get_vision_config(vision_model: str) -> dict:
    if vision_model not in VISION_REGISTRY:
        raise KeyError(
            f"Unknown vision model {vision_model!r}. "
            f"Known: {list(VISION_REGISTRY)}"
        )
    return VISION_REGISTRY[vision_model]
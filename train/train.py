import sys
import os
import math
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch.utils.data import DataLoader
from PIL import Image
import torch
import torch.nn as nn
from transformers import CLIPVisionModel, CLIPImageProcessor
from transformer_lens import HookedTransformer
from tqdm import tqdm
import torch.optim as optim
import wandb
from llava_dataset import LLaVADataset
import argparse
from accelerate import Accelerator
from accelerate.utils import set_seed
from sae_lens import SAE
from torch.utils.checkpoint import checkpoint
import re
from transformers import AutoImageProcessor, AutoModel, AutoTokenizer
from transformers import IJepaModel, AutoProcessor
import traceback
from transformers import get_cosine_schedule_with_warmup

# Set environment variables
# os.environ['WANDB_API_KEY'] = ''
# os.environ["HF_TOKEN"] = ""


def get_default_config():
    """Returns a dictionary with default configuration parameters"""
    return {
        # Training parameters
        'batch_size': 2,
        'n_batches': 100000000,
        'n_epochs': 1,
        'initial_learning_rate': 1e-3,
        'min_learning_rate': 2e-5,
        'warmup_ratio': 0.03,
        'name': 'pretrain-deepspeed',
        'json_file': '../data/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json',
        'image_dir': '../data/LLaVA-Pretrain/images/',
        'language_model': 'gemma-2-2b-it',
        'd_type': 'bfloat16',
        'max_length': 2048,
        'save_every': 100,
        'gradient_accumulation_steps': 32,
        'nsaes' : 0,
        'sae_layers': [], 
        'sae_constraints': 1,
        'vision_model' : 'clip'
    }

# %%
def get_args():
    parser = argparse.ArgumentParser()
    defaults = get_default_config()
    parser.add_argument('--sae_layer_str', type=str, default=None, help="Custom string for the SAE part of the filename.")
    for key, value in defaults.items():
        if key == 'sae_layers':
            parser.add_argument(f'--{key}', type=int, nargs='+', default=value)
        else:
            parser.add_argument(f'--{key}', type=type(value), default=value)
    
    args, unknown = parser.parse_known_args()
    return vars(args)

def get_config():
    if any(x in sys.modules for x in ['ipykernel', 'IPython']):
        return get_default_config()
    else:
        return get_args()

config = get_config()

SAE_LAYERS = config['sae_layers'] if config['sae_layers'] else list(range(config['nsaes']))
N_SAES = len(SAE_LAYERS) 
config['N_SAES'] = N_SAES
config['SAE_LAYERS'] = SAE_LAYERS
config['server'] = 'lambda'
config['n_gpus'] = str(torch.cuda.device_count())


IS_LLAMA = "llama" in config['language_model'].lower()
IS_GEMMA = "gemma" in config['language_model'].lower()

# Initialize accelerator with gradient accumulation
accelerator = Accelerator(gradient_accumulation_steps=config['gradient_accumulation_steps'])

device = accelerator.device
set_seed(475)

if accelerator.is_main_process:
    if accelerator.state.deepspeed_plugin is not None:
        print("✅ DeepSpeed is successfully enabled and configured.")
        print(f"   - ZeRO Stage: {accelerator.state.deepspeed_plugin.zero_stage}")
    else:
        print("❌ DeepSpeed is NOT active.")

# Define the path for saving and loading the projector weights
if config['sae_layer_str'] is not None:
    sae_layer_str = config['sae_layer_str']
else:
    sae_layer_str = "-".join(map(str, SAE_LAYERS)) if SAE_LAYERS else str(config['nsaes'])
    
safe_model_name = config["language_model"].replace("/", "_")

PROJECTOR_WEIGHTS_PATH = f'weights/projector_{safe_model_name}_{config["vision_model"]}_{config["name"]}_saes_{sae_layer_str}'

# Initialize models and processors on the CPU
device_map = {"": "cpu"} # A device_map to force loading on CPU. It will be loaded into GPU later by accelearte

if config['vision_model'] == 'clip':
    vision_model = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14", device_map='cpu')
    image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
elif config['vision_model'] == 'dinov2':
    vision_model = AutoModel.from_pretrained("facebook/dinov2-large")
    image_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-large")
elif config['vision_model'] == 'jepa':
    vision_model = IJepaModel.from_pretrained('facebook/ijepa_vith14_1k', device_map='cpu')
    image_processor = AutoProcessor.from_pretrained('facebook/ijepa_vith14_1k')

language_model = HookedTransformer.from_pretrained_no_processing(
    config['language_model'],
    dtype=config['d_type'],
    device='cpu'
)
tokenizer = language_model.tokenizer


criterion = nn.CrossEntropyLoss(ignore_index=-100)

class LLaVAModel(nn.Module):
    def __init__(self, vision_model, language_model, tokenizer, projection_dim=2304):
        super().__init__()
        self.projection_dim = language_model.cfg.d_model
        self.vision_model = vision_model
        self.language_model = language_model
        self.tokenizer = tokenizer
        self.projection = self.build_vision_projector(
            vision_model.config.hidden_size,
            self.projection_dim,
        )

        for param in self.vision_model.parameters():
            param.requires_grad = False
        for param in self.language_model.parameters():
            param.requires_grad = False

    def build_vision_projector(self, input_dim, output_dim):
        return nn.Linear(input_dim, output_dim)

    def load_projector_weights(self, path):
        path_ = path + '.pth'
        if os.path.exists(path_):
            d = torch.load(path_, map_location='cpu')
            if('module.weight' in d):
                d['weight'] = d['module.weight']
                d['bias'] = d['module.bias']
                del d['module.weight']
                del d['module.bias']
            self.projection.load_state_dict(d)
            print(f"Loaded projector weights from {path_}")
        else:
            if 'finetune' in path:
                raise(f"No projector weights found at {path_}")
            print(f"No projector weights found at {path_}")

    def save_projector_weights(self, path):
        torch.save(self.projection.state_dict(), path)

    def forward(self, image, input_ids, attention_mask, return_residuals=False):
        vision_outputs = self.vision_model(image)

        if config['vision_model'] in ['clip', 'dinov2']:
            image_features = self.projection(vision_outputs.last_hidden_state[:,1:])
        elif config['vision_model'] in ['jepa']:
            image_features = self.projection(vision_outputs.last_hidden_state)
        
        input_embeds = self.language_model.embed(input_ids)
        insert_position = 1

        prefix_embeds = input_embeds[:, :insert_position]
        suffix_embeds = input_embeds[:, insert_position:]
        combined_embeds = torch.cat([prefix_embeds, image_features, suffix_embeds], dim=1)
        
        prefix_attention = attention_mask[:, :insert_position]
        suffix_attention = attention_mask[:, insert_position:]
        vision_attention = torch.ones(attention_mask.shape[0], image_features.shape[1], device=attention_mask.device)
        combined_attention_mask = torch.cat([prefix_attention, vision_attention, suffix_attention], dim=1).to(torch.int64)

        x = combined_embeds
        residuals = {}

        num_visual_tokens = image_features.shape[1]
        
        for i, block in enumerate(self.language_model.blocks):
            x = block(x, attention_mask=combined_attention_mask)
            if return_residuals and i in SAE_LAYERS:
                residuals[i] = x[:, 1 : 1 + num_visual_tokens]

        x = self.language_model.ln_final(x)
        logits = x @ self.language_model.W_U
        
        if return_residuals:
            return logits, residuals
        return logits

# Create LLaVA model
llava_model = LLaVAModel(vision_model, language_model, tokenizer)
llava_model.load_projector_weights(PROJECTOR_WEIGHTS_PATH)
# NOTE: No .to(device) here, accelerate handles it.

dataset = LLaVADataset(json_file=config['json_file'],
                            image_dir=config['image_dir'],
                            tokenizer=tokenizer,
                            image_processor=image_processor,
                            model_type = 'llama' if IS_LLAMA else 'gemma')


def collate_fn(batch):
    images = torch.cat([item['image_tensor'] for item in batch], dim=0)
    
    # 1. Get raw data
    sequences = [item['input_ids'] for item in batch]
    assistant_positions = [item['assistant_positions'] for item in batch]
    
    # 2. Calc lengths and max_len
    # We need explicit lengths for the mask fix
    lengths = torch.tensor([seq.size(0) for seq in sequences])
    max_len = min(lengths.max().item(), config['max_length'])
    
    # Clamp lengths to max_len (handles truncation logic)
    lengths = lengths.clamp(max=max_len)

    # 3. Stack & Pad 
    # Ensure pad_token_id is safe (Llama 3 usually needs eos_token_id if pad is None)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    
    padded_sequences = torch.stack([
        torch.cat([
            seq[:max_len], 
            torch.full((max(0, max_len - seq.size(0)),), pad_id, dtype=torch.long)
        ]) for seq in sequences
    ])
    
    # 4. Vectorized Length-Based Masking
    # Creates a mask like [[1, 1, 1, 0], [1, 1, 0, 0]] purely based on size, ignoring token values.
    # [1, max_len] < [batch, 1] -> Broadcasts to [batch, max_len]
    attention_mask = (torch.arange(max_len).unsqueeze(0) < lengths.unsqueeze(1)).long()

    # 5. Handle Positions 
    truncated_positions = []
    for positions in assistant_positions:
        valid_positions = []
        for start, end in positions:
            if start < max_len - 1:
                valid_positions.append((start, min(end, max_len)))
        truncated_positions.append(valid_positions)

    return {
        "image_tensor": images,
        "input_ids": padded_sequences,
        "attention_mask": attention_mask,
        "assistant_positions": truncated_positions,
        "image_idx": [item['image_idx'] for item in batch],
    }

dataloader = DataLoader(
    dataset,
    batch_size=config['batch_size'],
    collate_fn=collate_fn,
    shuffle=True,
    drop_last=True
)

# Initialize optimizer, scheduler, and loss function
if accelerator.is_main_process:
    wandb.init(project="gemma_clip", config=config)
    run_name = wandb.run.name if wandb.run else "offline_run"
    # to save intermediate checkpoints
    save_dir = os.path.join("weights", run_name)
    os.makedirs(save_dir, exist_ok=True)
    print("run_name", run_name)

# 1. Setup Model & Optimizer
optimizer = optim.AdamW(llava_model.projection.parameters(), lr=config['initial_learning_rate'])

# 2. Prepare Data & Model FIRST (Do not pass scheduler yet)
llava_model, optimizer, dataloader = accelerator.prepare(
    llava_model, optimizer, dataloader
)

# 3. Calculate Steps using the PREPARED dataloader
num_update_steps_per_epoch = len(dataloader) // config['gradient_accumulation_steps']
total_training_steps = config['n_epochs'] * num_update_steps_per_epoch

# Apply your manual limit (if any)
if config['n_batches'] < total_training_steps:
    total_training_steps = config['n_batches']

print(f"✅ Rank {accelerator.process_index} Steps: {total_training_steps}")

# 4. Initialize Scheduler with the Precise Count
warmup_steps = int(total_training_steps * config['warmup_ratio'])

scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_training_steps
)

# 5. Register Scheduler manually so checkpoints save/load it correctly
accelerator.register_for_checkpointing(scheduler)


if accelerator.is_main_process:
    ds_config = accelerator.state.deepspeed_plugin.deepspeed_config
    print(f"✅ DeepSpeed Config Check:")
    print(ds_config)


if IS_GEMMA:
    model_param = re.findall(r"\db", config["language_model"])[0]
elif IS_LLAMA:
    model_param = '8b'

if config['sae_constraints']:
    saes = {}
    for layer in tqdm(SAE_LAYERS, desc="Loading SAEs"):
        print(f"LOADING layers: {layer} of SAE") 
        if IS_GEMMA:
            sae, _, _ = SAE.from_pretrained(
                # release="gemma-scope-2b-pt-res-canonical",
                release=f"gemma-scope-{model_param}-pt-res-canonical",
                sae_id=f"layer_{layer}/width_16k/canonical",
                device=device
            )
        elif IS_LLAMA:
            # For LLaVA 8B 
            # https://github.com/decoderesearch/SAELens/blob/v5.4.1/sae_lens/pretrained_saes.yaml#L10986
            sae, _, _ = SAE.from_pretrained(
                # release="gemma-scope-2b-pt-res-canonical",
                release=f"llama_scope_lxr_8x",
                sae_id=f"l{layer}r_8x",
                device='cpu'
            )
            sae.to(device)
            
            
        sae.to(dtype=torch.bfloat16)
        for param in sae.parameters():
            param.requires_grad = False
        saes[layer] = sae

# ==============================================================================
# Training Loop
# ==============================================================================
llava_model.train()
global_step = 0
for epoch in range(config['n_epochs']):
    print("Epoch Start", epoch)
    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch+1}/{config['n_epochs']}", disable=not accelerator.is_main_process)
    
    for step, batch in progress_bar:
        if step >= config['n_batches']:
            break
        
        skip_batch = torch.tensor(0, device=device)
        
        try:
            image = batch["image_tensor"]
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            assistant_positions = batch["assistant_positions"]
            
            # Forward pass
            if config['sae_constraints']:
                outputs, residuals = llava_model(image, input_ids, attention_mask, return_residuals=True)
            else:
                outputs = llava_model(image, input_ids, attention_mask)
            
            vision_token_offset = outputs.shape[1] - input_ids.shape[1]
            
            # Calculate loss
            batch_loss = torch.tensor(0.0, device=device) 
            batch_loss_sae = torch.tensor(0.0, device=device)
            valid_samples = 0
            
            for b in range(outputs.shape[0]):
                if len(assistant_positions[b]) == 0:
                    skip_batch += 1
                    continue
                
                sample_loss = torch.tensor(0.0, device=device)
                for start_pos, end_pos in assistant_positions[b]:
                    if start_pos >= end_pos:
                        continue
                    response_logits = outputs[b, vision_token_offset+start_pos-1:vision_token_offset+end_pos-1, :]
                    response_targets = input_ids[b, start_pos:end_pos]
                    loss = criterion(
                        response_logits.contiguous().view(-1, outputs.size(-1)),
                        response_targets.contiguous().view(-1)
                    )
                    sample_loss += loss / len(assistant_positions[b])
                
                batch_loss = batch_loss + sample_loss
                valid_samples += 1
            
            if config['sae_constraints']:
                for r_idx in SAE_LAYERS:
                    sae = saes[r_idx]
                    image_res_batch = residuals[r_idx]
                    sae_image_recon = sae.decode(sae.encode(image_res_batch))
                    batch_loss_sae = batch_loss_sae + (sae_image_recon - image_res_batch).pow(2).mean()
            
            # Normalize losses
            if valid_samples > 0:
                batch_loss = batch_loss / valid_samples
            if config['sae_constraints'] and N_SAES > 0:
                batch_loss_sae = batch_loss_sae / N_SAES
            
            sae_loss_weight = 1.
            
            # RAW LOSS (For Logging)
            batch_loss_total = batch_loss + sae_loss_weight * batch_loss_sae
            
            if batch_loss_total.dim() > 0:
                batch_loss_total = batch_loss_total.mean()

            # Create a new variable so we don't mess up the logging variable
            scaled_loss = batch_loss_total
            if config['gradient_accumulation_steps'] > 1:
                scaled_loss = scaled_loss / config['gradient_accumulation_steps']
            
            # Check for invalid loss
            if torch.isnan(scaled_loss) or torch.isinf(scaled_loss):
                dummy_loss = (outputs.sum() + residuals[SAE_LAYERS[0]].sum() if config['sae_constraints'] else outputs.sum()) * 0.0
                accelerator.backward(dummy_loss)
                skip_batch += 1
            elif scaled_loss.item() < 1e-10:
                dummy_loss = outputs.sum() * 0.0
                accelerator.backward(dummy_loss)
                skip_batch += 1
            else:
                accelerator.backward(scaled_loss)
            
            optimizer.step()
            optimizer.zero_grad()
                
        except Exception as e:
            print(f"Rank {accelerator.process_index} Error at step {step}: {e}")
            try:
                dummy_loss = llava_model.projection.weight.sum() * 0.0
                accelerator.backward(dummy_loss)
                optimizer.step()
                optimizer.zero_grad()
            except:
                pass
            skip_batch += 1
    
        # Sync skip flag across all GPUs
        skip_batch = accelerator.reduce(skip_batch, reduction="sum")
        
        # Logging and scheduling - only skip these if there was an error
        is_gradient_sync_step = ((step + 1) % config['gradient_accumulation_steps'] == 0)
        if is_gradient_sync_step:

            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            
            if accelerator.is_main_process:
                step_logs = {
                    "batch_loss_total": batch_loss_total.item(), 
                    "learning_rate": current_lr,
                    "batch_loss_sae": batch_loss_sae.item(),
                    "batch_loss": batch_loss.item(),
                    "global_step": global_step,
                }
                wandb.log(step_logs)
                progress_bar.set_postfix({"Total Loss": f"{batch_loss_total.item():.4f}", "LR": f"{current_lr:.6f}"})
            
            global_step += 1
            
            if global_step % config['save_every'] == 0:
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    unwrapped_model = accelerator.unwrap_model(llava_model)
                    unwrapped_model.save_projector_weights(PROJECTOR_WEIGHTS_PATH + ".pth")
                    save_filename = f"projector_{run_name}.pth"
                    full_save_path = os.path.join(save_dir, save_filename)
                    unwrapped_model.save_projector_weights(full_save_path)
                    wandb.save(PROJECTOR_WEIGHTS_PATH + ".pth")
        

# Final save
accelerator.wait_for_everyone()
if accelerator.is_main_process:
    unwrapped_model = accelerator.unwrap_model(llava_model)
    unwrapped_model.save_projector_weights(PROJECTOR_WEIGHTS_PATH + ".pth")
    save_filename = f"projector_{run_name}.pth"
    full_save_path = os.path.join(save_dir, save_filename)
    unwrapped_model.save_projector_weights(full_save_path)
    wandb.save(PROJECTOR_WEIGHTS_PATH + ".pth")
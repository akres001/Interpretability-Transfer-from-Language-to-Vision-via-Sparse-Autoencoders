import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import os
from PIL import ImageFile
import copy
import re  
import math 
import random
ImageFile.LOAD_TRUNCATED_IMAGES = True


class CacheDataset(Dataset):
    """LLaVA conversation dataset that formats prompts for either Gemma or Llama-3.
 
    Returns the first user turn + first assistant turn only, with positions
    indicating where the assistant response sits in the tokenized stream.
    """
 
    def __init__(self, json_file, image_dir, tokenizer, image_processor,
                 model_type="gemma", target_resolution=(224, 224)):
        with open(json_file, "r") as f:
            data = json.load(f)
        # Only keep examples that have an image attached
        self.data = [item for item in data if "image" in item]
 
        self.image_dir = image_dir
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.target_resolution = target_resolution
        self.model_type = model_type.lower()
 
        print(f"Loaded {len(self.data)} examples for model: {self.model_type}")
 
        if self.model_type == "llama":
            self.sep_user = "<|start_header_id|>user<|end_header_id|>\n\n"
            self.sep_model = "<|start_header_id|>model<|end_header_id|>\n\n"
            self.sep_end = "<|eot_id|>"
        else:  # gemma
            self.sep_user = "<start_of_turn>user\n"
            # Gemma merges the end of the user turn with the start of the model turn
            self.sep_user_to_model = "<end_of_turn>\n<start_of_turn>model\n"
            self.sep_end = "<end_of_turn>\n"
 
    def __len__(self):
        return len(self.data)
 
    @staticmethod
    def _ensure_image_tag(text):
        """Place a single <image> tag at the start of the user message."""
        text = text.replace("<image>", "")
        return "<image>\n" + text
 
    def _token_len(self, text):
        return len(self.tokenizer(text, return_tensors="pt")["input_ids"][0])
 
    def _append_user(self, conv_so_far, content):
        """Append a user turn and return the new conversation string."""
        content = self._ensure_image_tag(content)
        if self.model_type == "llama":
            return conv_so_far + self.sep_user + content + self.sep_end
        # Gemma combines end-of-user with start-of-model in a single separator
        return conv_so_far + self.sep_user + content + self.sep_user_to_model
 
    def _append_assistant(self, conv_so_far, content):
        """Append an assistant turn; return (new_conv, (start_pos, end_pos))."""
        if self.model_type == "llama":
            # Llama needs the assistant header explicitly
            conv_so_far = conv_so_far + self.sep_model
 
        start_pos = self._token_len(conv_so_far)
        end_pos = self._token_len(conv_so_far + content)
        conv_so_far = conv_so_far + content + self.sep_end
        # +1 to include the EOS/end-of-turn token
        return conv_so_far, (start_pos, end_pos + 1)
 
    def __getitem__(self, idx):
        try:
            item = self.data[idx]
            image_path = os.path.join(self.image_dir, item["image"])
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {idx}: {e}")
            return self.__getitem__(random.choice(range(len(self.data))))
 
        image_tensor = self.image_processor(
            images=image, return_tensors="pt"
        )["pixel_values"]
  
        # Build only the first user->assistant turn (the rest of the conversation
        # is not needed for activation caching).
        full_conversation = ""
        assistant_positions = []
 
        for conv in item["conversations"]:
            conv_copy = copy.deepcopy(conv)
            if conv_copy["from"] == "human":
                full_conversation = self._append_user(full_conversation, conv_copy["value"])
            else:
                full_conversation, span = self._append_assistant(
                    full_conversation, conv_copy["value"]
                )
                assistant_positions.append(span)
 
            # Only keep the first assistant turn
            if len(assistant_positions) == 1:
                break
 
        conversation_encoding = self.tokenizer(full_conversation, return_tensors="pt")
        return {
            "image_idx": item["image"],
            "image_path": item['image'],
            "image_tensor": image_tensor,
            "input_ids": conversation_encoding["input_ids"].squeeze(),
            "attention_mask": conversation_encoding["attention_mask"].squeeze(),
            "assistant_positions": assistant_positions,
        }



class GemmaLLaVADatasetSimplified(Dataset):
    def __init__(self, json_file, image_dir, tokenizer, image_processor, max_length=2048):
        """
        Dataset for LLaVA-style data with Gemma model compatibility
        
        Args:
            json_file: Path to JSON file containing annotations
            image_dir: Directory where images are stored
            tokenizer: Tokenizer for the language model
            image_processor: Processor for CLIP images
            max_length: Maximum sequence length for tokens
        """
        self.image_dir = image_dir
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_length = max_length
        
        # Load annotations from JSON file
        with open(json_file, 'r') as f:
            self.annotations = json.load(f)
        # Filter out invalid entries
        self.annotations_temp = [ann for ann in self.annotations if self._validate_annotation(ann)]

        self.annotations = []
        for ii, el in enumerate(self.annotations_temp):
            el['unique_id'] = ii
            self.annotations.append(el)
            
    
    def _validate_annotation(self, annotation):
        """Validate that annotation has required fields"""
        return ('image' in annotation and 
                'conversations' in annotation and 
                len(annotation['conversations']) >= 2)
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        
        # Load and process image
        image_path = os.path.join(self.image_dir, annotation['image'])
        image = Image.open(image_path)
        
        return {
            'image_id' : annotation['image'],
            'id' : annotation['id'],
            'unique_id' : annotation['unique_id'],
            'image': image,
        }
 
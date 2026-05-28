import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import os
from PIL import ImageFile
import random
ImageFile.LOAD_TRUNCATED_IMAGES = True


class LLaVADataset(Dataset):
    def __init__(self, json_file, image_dir, tokenizer, image_processor, model_type="gemma", target_resolution=(224, 224)):
        self.offset = 0
        
        with open(json_file, 'r') as f:
            data = json.load(f)
            data = [item for item in data if 'image' in item]
        
        self.data = data
        self.image_dir = image_dir
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.target_resolution = target_resolution
        self.model_type = model_type.lower()
        
        print(f"USING Dataset for model: {self.model_type}")

        # --- Define Formatting Strings ---
        if self.model_type == "llama":
            # Llama 3 / LLaVA 3.1 Format
            self.sep_user = "<|start_header_id|>user<|end_header_id|>\n\n"
            self.sep_model = "<|start_header_id|>model<|end_header_id|>\n\n"
            self.sep_end = "<|eot_id|>"
            self.bos = "<|begin_of_text|>"
        else:
            # Gemma 2 Format (Default)
            self.sep_user = "<start_of_turn>user\n"
            self.sep_model = "<end_of_turn>\n<start_of_turn>model\n" 
            self.sep_end = "<end_of_turn>\n"
            self.bos = "<bos>" # Gemma usually handles bos automatically, but we can track it

    def __len__(self):
        return len(self.data)
    
    def get_image(self, idx):
        if 'image' in self.data[idx]:
            image_path = f"{self.image_dir}{self.data[idx]['image']}"
            return Image.open(image_path).convert("RGB")
        return None

    def expand2square(self,pil_img, background_color):
        width, height = pil_img.size
        if width == height:
            return pil_img
        elif width > height:
            result = Image.new(pil_img.mode, (width, width), background_color)
            result.paste(pil_img, (0, (width - height) // 2))
            return result
        else:
            result = Image.new(pil_img.mode, (height, height), background_color)
            result.paste(pil_img, ((height - width) // 2, 0))
            return result

    def __getitem__(self, idx):
        try:
            item = self.data[idx]
            image = self.get_image(idx)
        except Exception as e:
            print(f"Error loading image {idx}: {e}")
            return self.__getitem__(random.choice(range(len(self.data))))
        
        
        if image is not None:
            image_tensor = self.expand2square(image, tuple(int(x*255) for x in self.image_processor.image_mean))
            image_tensor = self.image_processor.preprocess(image_tensor, return_tensors='pt')['pixel_values']
        else:
            return None


        # --- Conversation Processing ---
        
        full_conversation = ""

        assistant_positions = []
        
        for i, conv in enumerate(item['conversations']):
            # Ensure <image> token exists for the first user turn (Required for LLaVA)
            val = conv['value']
            # if i == 0 and conv['from'] == 'human' and "<image>" not in val:
            #     val = "<image>\n" + val

            if conv['from'] == 'human':
             
                if self.model_type == "llama":
                    # Llama doesn't merge the "end of previous" with "start of current" like your original code
                    # Your original code: turn_end = "<end_of_turn>\n<start_of_turn>model\n"
                    # We will follow the distinct structure:
                    full_conversation += self.sep_user + val + self.sep_end
                else:
                    # Gemma logic from your snippet
                    full_conversation += "<start_of_turn>user\n" + val + "<end_of_turn>\n<start_of_turn>model\n"

            else:
                # Add assistant part
                response_content = conv['value']
                
                if self.model_type == "llama":
                    # For Llama, we must append the assistant header first
                    full_conversation += self.sep_model
                    
                    # Track Start
                    start_pos = len(self.tokenizer(full_conversation, return_tensors="pt")['input_ids'][0])
                    
                    # Track End (Temp)
                    temp_conv = full_conversation + response_content
                    end_pos = len(self.tokenizer(temp_conv, return_tensors="pt")['input_ids'][0])
                    
                    # Add content + EOT
                    full_conversation += response_content + self.sep_end
                
                else:
                    # Original Gemma Logic (Header was added at end of previous user block)
                    start_pos = len(self.tokenizer(full_conversation, return_tensors="pt")['input_ids'][0])
                    
                    temp_conv = full_conversation + response_content
                    end_pos = len(self.tokenizer(temp_conv, return_tensors="pt")['input_ids'][0])
                    
                    full_conversation += response_content + "<end_of_turn>\n"

                # + 1 to include eos token
                assistant_positions.append((start_pos, end_pos + 1)) 

        # Tokenize full conversation
        conversation_encoding = self.tokenizer(full_conversation, return_tensors="pt")
        
        return {
            "image_idx": idx,
            "image_tensor": image_tensor,
            "input_ids": conversation_encoding["input_ids"].squeeze(),
            "attention_mask": conversation_encoding["attention_mask"].squeeze(),
            "assistant_positions": assistant_positions
        }
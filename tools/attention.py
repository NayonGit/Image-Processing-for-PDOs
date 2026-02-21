import gc
import h5py
import math
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import torch
import torch.nn as nn


class AttentionVisualizer:
    def __init__(self, device='cpu'):
        """
        Toolbox to extract and visualize self-attention maps.
        Unified logic for DINOv2/v3 (Meta) and Curia (HF).
        """
        self.device = device

    def _get_img_from_h5(self, h5_path, img_index):
        if not os.path.exists(h5_path):
            raise FileNotFoundError(f"H5 file not found at {h5_path}")
        with h5py.File(h5_path, 'r') as hdf:
            group = hdf['train']['images']
            img_names = list(group.keys())
            return np.array(group[img_names[img_index]])

    def get_attention_maps(self, model, transform, h5_path, img_index=0, block_index=11, img_size=448):
        model.to(self.device)
        model.eval()

        
        is_curia = hasattr(model, 'encoder')
        is_dinov3 = hasattr(model,'rope_embed')
        
        # Color conversion must match specific model requirements
        # Image Loading 
        img_raw = self._get_img_from_h5(h5_path, img_index)
        color_mode = 'L' if is_curia else 'RGB'
        img_pil = Image.fromarray(img_raw).convert(color_mode)
        img_display = img_pil.convert('RGB')
        img_tensor = transform(img_pil).unsqueeze(0).to(self.device)

        attention_probs = []

        def hook_fn(module, input, output):
    
            x = input[0] 
            B, N, C = x.shape
            
            # DINOv3
            # Due to a problem with a "rope" function applied to q and k, we cannot reliably extract attention maps from the 'attn' module.
            # We modified the layers to add an option to calculate by hand the attention of the desired module.
            if hasattr(module, 'save_attn') and module.save_attn:
                attn = module.last_attn
                if attn is not None:
                    attention_probs.append(attn.detach().cpu())
                else:
                    print("Warning: Attention map not available for this block.")
            # Curia 
            elif hasattr(module, 'query'):
                num_heads = module.num_attention_heads
                head_dim = module.attention_head_size
                q = module.query(x).view(B, N, num_heads, head_dim).transpose(1, 2)
                k = module.key(x).view(B, N, num_heads, head_dim).transpose(1, 2)
                scale = head_dim ** -0.5
                attn = (q @ k.transpose(-2, -1)) * scale
                attn = attn.softmax(dim=-1)
                attention_probs.append(attn.detach().cpu())
                
            # DINOv2 
            elif hasattr(module, 'qkv') and not is_dinov3:
                num_heads = module.num_heads
                head_dim = C // num_heads
                
                # We apply the norm here because the hook on 'attn' receives normed input
                qkv = module.qkv(x).reshape(B, N, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
                q, k = qkv[0], qkv[1]
                
                scale = getattr(module, 'scale', None) or (head_dim ** -0.5)
                attn = (q @ k.transpose(-2, -1)) * scale
                attn = attn.softmax(dim=-1)
                attention_probs.append(attn.detach().cpu())

            

        # Target selection
        if is_curia:
            target_layer = model.encoder.layer[block_index].attention.attention
        elif is_dinov3:
            target_layer = model.blocks[block_index].attn
            model.blocks[block_index].attn.save_attn = True
        else:
            target_layer = model.blocks[block_index].attn

        handle = target_layer.register_forward_hook(hook_fn)

        try:
            with torch.inference_mode():
                model(img_tensor)
            
            attentions = attention_probs[0][0]
            nh = attentions.shape[0]
            total_tokens = attentions.shape[-1]
            side = int(math.sqrt(total_tokens - 1))
            
            # Extracting patches (ignoring CLS and potential registers)
            num_patches = side * side
            map_att = attentions[:, 0, -num_patches:].reshape(nh, side, side)

            # Plotting
            fig, axes = plt.subplots(1, nh + 1, figsize=(22, 5))
            axes[0].imshow(img_display.resize((img_size, img_size)))
            axes[0].set_title(f"Original\nIdx {img_index}, Blk {block_index}")
            axes[0].axis('off')
            
            for i in range(nh):
                axes[i+1].imshow(map_att[i].numpy(), cmap='magma', interpolation='nearest')
                axes[i+1].set_title(f"Head {i}")
                axes[i+1].axis('off')
            
            plt.tight_layout()
            plt.show()

        finally:
            handle.remove()



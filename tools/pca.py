import gc
import h5py
import math
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
from sklearn.decomposition import PCA
import torch
import torch.nn as nn

class PCAVisualizer:
    def __init__(self, device='cpu'):
        """
        Toolbox for PCA-based semantic segmentation of ViT patch embeddings.
        Supports: DINOv2, DINOv3, and Curia.
        """
        self.device = device

    def _get_img_from_h5(self, h5_path, img_index):
        if not os.path.exists(h5_path):
            raise FileNotFoundError(f"H5 file not found at {h5_path}")
        with h5py.File(h5_path, 'r') as hdf:
            group = hdf['train']['images']
            img_names = list(group.keys())
            return np.array(group[img_names[img_index]])

    def _interpolate_pos_encoding(self, model, x, w, h):
        """Adapts learned position embeddings to new image sizes (DINOv2/v3)."""
        if not hasattr(model, 'pos_embed'):
            return 0 # Not needed for Curia as it's handled in embeddings forward
            
        npatch = x.shape[1] - 1
        N = model.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return model.pos_embed
        
        class_pos_embed = model.pos_embed[:, 0]
        patch_pos_embed = model.pos_embed[:, 1:]
        dim = x.shape[-1]
        
        w0 = h0 = int(math.sqrt(N))
        # Logic for 14x14 patches (DINO standard)
        patch_pos_embed = torch.nn.functional.interpolate(
            patch_pos_embed.reshape(1, w0, h0, dim).permute(0, 3, 1, 2),
            size=(w // 14, h // 14),
            mode='bicubic',
            align_corners=False,
        )
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def get_pca_maps(self, model, transform, h5_path, img_index=0, block_index=11, 
                     img_size=448, bg_threshold=0.3, invert_pc=False):
        """
        Main pipeline to extract embeddings, run PCA, and display RGB semantic maps.
        """
        model.to(self.device)
        model.eval()

        # Image Loading
        img_raw = self._get_img_from_h5(h5_path, img_index)
        is_curia = hasattr(model, 'encoder')
        
        color_mode = 'L' if is_curia else 'RGB'
        img_pil = Image.fromarray(img_raw).convert(color_mode)
        img_display = img_pil.convert('RGB')
        img_tensor = transform(img_pil).unsqueeze(0).to(self.device)

        # Embedding Extraction
        with torch.no_grad():
            if is_curia:
                # Curia logic
                outputs = model(img_tensor, output_hidden_states=True)
                x = outputs.hidden_states[block_index + 1]
            else:
                # DINOv2 / DINOv3 logic
                x = model.patch_embed(img_tensor)
                if x.ndim == 4: # DINOv3 sometimes returns [B, H, W, C]
                    B, H, W, C = x.shape
                    x = x.reshape(B, H * W, C)
                else:
                    B = x.shape[0]

                cls_token = model.cls_token.expand(x.shape[0], -1, -1).to(x.device)
                x = torch.cat((cls_token, x), dim=1)
                
                # Apply Positional Encoding if not handled by the forward
                x = x + self._interpolate_pos_encoding(model, x, img_size, img_size)
                
                # Pass through blocks up to target
                for i in range(block_index + 1):
                    x = model.blocks[i](x)

        # PCA Calculation
        patch_embeddings = x[0, 1:].cpu().numpy() # [N_patches, Dim]
        del x # Free memory
        gc.collect()

        pca = PCA(n_components=3)
        pca_features = pca.fit_transform(patch_embeddings)

        # Grid Reshaping & Padding Handling
        total_patches = pca_features.shape[0]
        grid_size = int(np.sqrt(total_patches))
        n_reg = total_patches - (grid_size * grid_size)
        
        if n_reg > 0: # Remove extra tokens (registers) from PCA features
            pca_features = pca_features[n_reg:]
            grid_size = int(np.sqrt(pca_features.shape[0]))

        # Normalization & Masking
        rgb_channels = []
        for i in range(3):
            pc = pca_features[:, i]
            pc_min, pc_max = pc.min(), pc.max()
            pc_norm = (pc - pc_min) / (pc_max - pc_min + 1e-8)
            if invert_pc:
                pc_norm = 1.0 - pc_norm
            rgb_channels.append(pc_norm)

        # Background Masking using PC1
        mask = rgb_channels[0] > bg_threshold
        rgb_flat = np.stack(rgb_channels, axis=-1)
        rgb_flat = rgb_flat * mask[:, np.newaxis]

        # Visualization
        pca_maps = rgb_flat.reshape(grid_size, grid_size, 3)
        
        fig, axes = plt.subplots(1, 5, figsize=(25, 5))
        axes[0].imshow(img_display.resize((img_size, img_size)))
        axes[0].set_title(f"Original\nIdx {img_index}, Blk {block_index}")
        axes[0].axis('off')

        titles = ["PC1 (R)", "PC2 (G)", "PC3 (B)"]
        for i in range(3):
            comp_map = rgb_channels[i].reshape(grid_size, grid_size)
            axes[i+1].imshow(comp_map, cmap='gray', interpolation='nearest', vmin=0, vmax=1)
            axes[i+1].set_title(f"{titles[i]}\n{pca.explained_variance_ratio_[i]*100:.1f}% var")
            axes[i+1].axis('off')

        axes[4].imshow(pca_maps, interpolation='nearest')
        axes[4].set_title(f"PCA RGB\nTotal: {pca.explained_variance_ratio_.sum()*100:.1f}%")
        axes[4].axis('off')

        plt.tight_layout()
        plt.show()

        print(f"Explained variance: {pca.explained_variance_ratio_.sum()*100:.2f}%")
        print(f"Foreground: {100*mask.sum()/mask.size:.1f}% of image")
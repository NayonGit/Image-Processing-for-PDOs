import math
import sys
from pathlib import Path
from types import SimpleNamespace
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics

from detection_dino_lora.lora import create_backbone_with_lora
from detection_dino_lora.utils import get_boundaries, extract_patch_targets

# ---- third_party DINO imports ----
DINO_REPO = Path(__file__).resolve().parents[1] / "third_party" / "DINO"
if str(DINO_REPO) not in sys.path:
    sys.path.insert(0, str(DINO_REPO))

from util.misc import NestedTensor
from util import box_ops
from models.dino.position_encoding import build_position_encoding
from models.dino.deformable_transformer import build_deformable_transformer
from models.dino.dino import DINO, SetCriterion, PostProcess
from models.dino.matcher import build_matcher


class LoRAJoinerForDINO(nn.Module):
    def __init__(self, foundation_backbone: nn.Module, hidden_dim: int = 256, num_feature_levels: int = 4):
        super().__init__()
        self.foundation_backbone = foundation_backbone
        in_c = foundation_backbone.num_features
        self.proj0 = nn.Sequential(nn.Conv2d(in_c, hidden_dim, kernel_size=1), nn.GroupNorm(32, hidden_dim))
        self.down_blocks = nn.ModuleList([
            nn.Sequential(nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1), nn.GroupNorm(32, hidden_dim))
            for _ in range(num_feature_levels - 1)
        ])
        self.num_channels = [hidden_dim] * num_feature_levels

    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors
        b = x.shape[0]
        tokens = self.foundation_backbone.extract_features(x)  # [B,N,C]
        n = tokens.shape[1]
        s = int(math.sqrt(n))
        if s * s != n:
            raise ValueError(f"Non-square token map: N={n}")

        feat = tokens.transpose(1, 2).reshape(b, -1, s, s)
        feat = self.proj0(feat)

        feats = [feat]
        for blk in self.down_blocks:
            feats.append(blk(feats[-1]))

        out = OrderedDict()
        for i, f in enumerate(feats):
            m = tensor_list.mask
            mask_i = F.interpolate(m[None].float(), size=f.shape[-2:], mode="nearest")[0].to(torch.bool)
            out[str(i)] = NestedTensor(f, mask_i)
        return out


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out, pos = [], []
        for _, x in xs.items():
            out.append(x)
            pos.append(self[1](x).to(x.tensors.dtype))
        return out, pos


class LoRADINOModel(L.LightningModule):
    def __init__(
        self,
        backbone_name: str = "dinov2",
        backbone_size: str = "small",
        num_classes: int = 1,
        hidden_dim: int = 256,
        num_queries: int = 100,
        nheads: int = 8,
        enc_layers: int = 6,
        dec_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.0,
        # LoRA
        lora_rank: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        lora_target_modules: Optional[List[str]] = None,
        use_dora: bool = False,
        # loss/matcher
        focal_alpha: float = 0.25,
        cost_class: float = 2.0,
        cost_bbox: float = 5.0,
        cost_giou: float = 2.0,
        cls_loss_coef: float = 1.0,
        bbox_loss_coef: float = 5.0,
        giou_loss_coef: float = 2.0,
        # dn
        use_dn: bool = True,
        dn_number: int = 100,
        dn_box_noise_scale: float = 0.4,
        dn_label_noise_ratio: float = 0.5,
        # optim
        lr: float = 1e-4,
        lr_backbone: float = 1e-5,
        weight_decay: float = 1e-4,
        # patching (identique stratégie legacy)
        use_patching: bool = False,
        img_size: int = 1024,
        num_patches: int | None = None,
        patch_size: int = 224,
        overlap_size: int = 30,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = not use_patching

        self.lr = lr
        self.lr_backbone = lr_backbone
        self.weight_decay = weight_decay
        self.use_patching = use_patching
        self.img_size = img_size
        self.patch_size = patch_size
        self.val_metric = torchmetrics.detection.MeanAveragePrecision(iou_type="bbox")

        if use_patching:
            self.boundaries, self.img_size = get_boundaries(num_patches, overlap_size, patch_size, img_size=self.img_size)
        else:
            self.boundaries = None

        foundation = create_backbone_with_lora(
            name=backbone_name,
            model_size=backbone_size,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=lora_target_modules,
            use_dora=use_dora,
        )

        args = SimpleNamespace(
            # --- Arguments de base ---
            device="cuda" if torch.cuda.is_available() else "cpu",
            num_classes=num_classes,
            num_queries=num_queries,
            hidden_dim=hidden_dim,
            nheads=nheads,
            enc_layers=enc_layers,
            dec_layers=dec_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,

            # --- Arguments réclamés par build_deformable_transformer ---
            decoder_layer_noise=False,     # Utilisé dans le premier IF
            dln_xy_noise=0.0,              # Requis si decoder_layer_noise est True (on met 0 par sécu)
            dln_hw_noise=0.0,              # Requis si decoder_layer_noise est True
            
            unic_layers=0,                 # Mappe vers num_unicoder_layers
            pre_norm=False,                # Mappe vers normalize_before
            query_dim=4,
            transformer_activation="relu",
            num_patterns=0,
            
            num_feature_levels=4,
            enc_n_points=4,
            dec_n_points=4,
            use_deformable_box_attn=False,
            box_attn_type='roi_align',
            
            add_channel_attention=False,
            add_pos_value=False,
            random_refpoints_xy=False,
            
            # --- Two Stage ---
            two_stage_type="no",  # "standard", "early", "enceachlayer", "enclayer1"
            two_stage_pat_embed=0,
            two_stage_add_query_num=0,
            two_stage_learn_wh=False,
            two_stage_keep_all_tokens=False,
            dec_layer_number=None,
            
            # --- Structure du Décodeur ---
            decoder_sa_type="sa",
            decoder_module_seq=['sa', 'ca', 'ffn'], # ATTENTION: s'appelle decoder_module_seq ici
            embed_init_tgt=False,
            use_detached_boxes_dec_out=False,

            # --- Arguments pour le reste du modèle (Matcher/Criterion) ---
            matcher_type="HungarianMatcher",
            dn_number=dn_number,
            use_dn=use_dn,
            dn_box_noise_scale=dn_box_noise_scale,
            dn_label_noise_ratio=dn_label_noise_ratio,
            dn_labelbook_size=max(2, num_classes + 1),
            set_cost_class=cost_class,
            set_cost_bbox=cost_bbox,
            set_cost_giou=cost_giou,
            focal_alpha=focal_alpha,
            # Ajoute les coefs de loss s'ils sont utilisés dans le Criterion
            cls_loss_coef=cls_loss_coef,
            bbox_loss_coef=bbox_loss_coef,
            giou_loss_coef=giou_loss_coef,
            # --- Position Embedding (Requis par le Joiner) ---
            position_embedding="sine",      # L'erreur actuelle
            pe_temperatureH=20,             # Température pour l'encodage vertical
            pe_temperatureW=20,             # Température pour l'encodage horizontal
            N_steps=hidden_dim // 2,        # Souvent requis pour diviser le d_model en X/Y
            
            # --- Backprop / Training ---
            masks=False,                    # On travaille sur des boîtes, pas des segmentations
            aux_loss=True,                  # Important pour DINO (calcule la loss à chaque couche)
        )
        
        position_embedding = build_position_encoding(args)
        wrapped_backbone = LoRAJoinerForDINO(foundation, hidden_dim=hidden_dim, num_feature_levels=4)
        backbone = Joiner(wrapped_backbone, position_embedding)
        backbone.num_channels = [hidden_dim] * 4

        transformer = build_deformable_transformer(args)
        self.model = DINO(
            backbone=backbone,
            transformer=transformer,
            num_classes=num_classes,
            num_queries=num_queries,
            aux_loss=True,
            iter_update=True,
            query_dim=4,
            random_refpoints_xy=False,
            fix_refpoints_hw=-1,
            num_feature_levels=4,
            nheads=nheads,
            two_stage_type="no",
            dec_pred_class_embed_share=True,
            dec_pred_bbox_embed_share=True,
            two_stage_class_embed_share=True,
            two_stage_bbox_embed_share=True,
            decoder_sa_type="sa",
            num_patterns=0,
            dn_number=dn_number if use_dn else 0,
            dn_box_noise_scale=dn_box_noise_scale,
            dn_label_noise_ratio=dn_label_noise_ratio,
            dn_labelbook_size=max(2, num_classes + 1),
        )

        matcher = build_matcher(args)
        weight_dict = {"loss_ce": cls_loss_coef, "loss_bbox": bbox_loss_coef, "loss_giou": giou_loss_coef}
        if use_dn:
            weight_dict.update({"loss_ce_dn": cls_loss_coef, "loss_bbox_dn": bbox_loss_coef, "loss_giou_dn": giou_loss_coef})
        for i in range(dec_layers - 1):
            for k, v in list(weight_dict.items()):
                weight_dict[f"{k}_{i}"] = v
        weight_dict.update({"loss_ce_interm": cls_loss_coef, "loss_bbox_interm": bbox_loss_coef, "loss_giou_interm": giou_loss_coef})

        self.criterion = SetCriterion(
            num_classes=num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            focal_alpha=focal_alpha,
            losses=["labels", "boxes", "cardinality"],
        )
        self.postprocessor = PostProcess(num_select=100, nms_iou_threshold=-1)

    def _abs_xyxy_to_norm_cxcywh(self, boxes_xyxy_abs: torch.Tensor, size_hw: torch.Tensor) -> torch.Tensor:
        h, w = size_hw[0].float(), size_hw[1].float()
        norm = torch.tensor([w, h, w, h], device=boxes_xyxy_abs.device).unsqueeze(0)
        boxes_xyxy_norm = boxes_xyxy_abs / norm
        return box_ops.box_xyxy_to_cxcywh(boxes_xyxy_norm).clamp(0, 1)

    def _prepare_targets_for_dino(self, targets: List[Dict], device: torch.device) -> List[Dict]:
        out = []
        for t in targets:
            boxes = t["boxes"].to(device).float()  # ABS XYXY
            labels = t["labels"].to(device).long()
            size = t["size"].to(device)
            out.append({"labels": labels, "boxes": self._abs_xyxy_to_norm_cxcywh(boxes, size)})
        return out

    def _compute_loss(self, outputs: Dict, dino_targets: List[Dict]) -> Tuple[torch.Tensor, Dict]:
        loss_dict = self.criterion(outputs, dino_targets)
        total = sum(loss_dict[k] * self.criterion.weight_dict[k] for k in loss_dict if k in self.criterion.weight_dict)
        return total, loss_dict

    def _update_metrics(self, outputs: Dict, raw_targets: List[Dict], device: torch.device, target_size_hw: torch.Tensor):
        bs = len(raw_targets)
        target_sizes = target_size_hw.unsqueeze(0).repeat(bs, 1).to(device)
        preds = self.postprocessor(outputs, target_sizes)

        pred_list, gt_list = [], []
        for p, t in zip(preds, raw_targets):
            pred_list.append({
                "boxes": p["boxes"].detach(),
                "scores": p["scores"].detach(),
                "labels": p["labels"].detach().to(torch.int64),
            })
            gt_list.append({
                "boxes": t["boxes"].to(device).float(),
                "labels": t["labels"].to(device).long(),
            })
        self.val_metric.update(pred_list, gt_list)

    def training_step(self, batch, batch_idx):
        

        images, targets = batch
        device = images.device

        if not self.use_patching:
            dino_targets = self._prepare_targets_for_dino(targets, device)
            outputs = self.model(images, dino_targets)
            total_loss, loss_dict = self._compute_loss(outputs, dino_targets)

            for k, v in loss_dict.items():
                self.log(f"train/{k}", v, on_epoch=True, on_step=False, batch_size=len(images))
            self.log("train/loss", total_loss, on_epoch=True, on_step=True, prog_bar=True, batch_size=len(images))
            return total_loss

        opt = self.optimizers()
        losses = []

        for start_y, end_y in self.boundaries:
            for start_x, end_x in self.boundaries:
                opt.zero_grad()
                patch = images[..., start_y:end_y, start_x:end_x]

                # targets déjà absolus en pixels sur image resized globale
                # extract_patch_targets attend boxes normalisées cxcywh dans ton implémentation actuelle;
                # donc on convertit localement avant et après.
                pseudo_targets = []
                for t in targets:
                    size = t["size"].to(device)
                    boxes_norm = self._abs_xyxy_to_norm_cxcywh(t["boxes"].to(device).float(), size)
                    pseudo_targets.append({"boxes": boxes_norm, "labels": t["labels"].to(device)})

                patch_bounds = (start_y, end_y, start_x, end_x)
                patch_targets_norm = extract_patch_targets(
                    pseudo_targets, patch_bounds, self.img_size, patch_size=self.patch_size
                )

                # repasse en abs xyxy patch-space pour garder pipeline unique
                patch_targets_abs = []
                for pt in patch_targets_norm:
                    boxes_xyxy_norm = box_ops.box_cxcywh_to_xyxy(pt["boxes"].to(device).float()).clamp(0, 1)
                    boxes_xyxy_abs = boxes_xyxy_norm * self.patch_size
                    patch_targets_abs.append({
                        "boxes": boxes_xyxy_abs,
                        "labels": pt["labels"].to(device).long(),
                        "size": torch.tensor([self.patch_size, self.patch_size], device=device),
                    })

                dino_patch_targets = []
                for pt in patch_targets_abs:
                    dino_patch_targets.append({
                        "labels": pt["labels"],
                        "boxes": self._abs_xyxy_to_norm_cxcywh(pt["boxes"], pt["size"]),
                    })

                outputs = self.model(patch, dino_patch_targets)
                total_loss, _ = self._compute_loss(outputs, dino_patch_targets)

                self.manual_backward(total_loss)
                self.clip_gradients(opt, gradient_clip_val=0.1, gradient_clip_algorithm="norm")
                opt.step()
                losses.extend([total_loss.detach()] * len(images))

        avg_loss = sum(losses) / len(losses)
        self.log("train/loss", avg_loss, on_epoch=True, on_step=True, prog_bar=True, batch_size=len(images))
        return avg_loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        device = images.device

        if not self.use_patching:
            dino_targets = self._prepare_targets_for_dino(targets, device)
            outputs = self.model(images, dino_targets)
            total_loss, _ = self._compute_loss(outputs, dino_targets)
            self.log("val/loss", total_loss, on_epoch=True, on_step=False, prog_bar=True, batch_size=len(images))
            self._update_metrics(outputs, targets, device, torch.tensor([self.img_size, self.img_size], device=device))
            return total_loss

        losses = []
        for start_y, end_y in self.boundaries:
            for start_x, end_x in self.boundaries:
                patch = images[..., start_y:end_y, start_x:end_x]

                pseudo_targets = []
                for t in targets:
                    size = t["size"].to(device)
                    boxes_norm = self._abs_xyxy_to_norm_cxcywh(t["boxes"].to(device).float(), size)
                    pseudo_targets.append({"boxes": boxes_norm, "labels": t["labels"].to(device)})

                patch_bounds = (start_y, end_y, start_x, end_x)
                patch_targets_norm = extract_patch_targets(
                    pseudo_targets, patch_bounds, self.img_size, patch_size=self.patch_size
                )

                patch_targets_abs = []
                for pt in patch_targets_norm:
                    boxes_xyxy_norm = box_ops.box_cxcywh_to_xyxy(pt["boxes"].to(device).float()).clamp(0, 1)
                    boxes_xyxy_abs = boxes_xyxy_norm * self.patch_size
                    patch_targets_abs.append({
                        "boxes": boxes_xyxy_abs,
                        "labels": pt["labels"].to(device).long(),
                        "size": torch.tensor([self.patch_size, self.patch_size], device=device),
                    })

                dino_patch_targets = [{"labels": pt["labels"], "boxes": self._abs_xyxy_to_norm_cxcywh(pt["boxes"], pt["size"])}
                                      for pt in patch_targets_abs]

                outputs = self.model(patch, dino_patch_targets)
                total_loss, _ = self._compute_loss(outputs, dino_patch_targets)
                losses.extend([total_loss.detach()] * len(images))

                raw_patch_targets_for_map = [{"boxes": pt["boxes"], "labels": pt["labels"]} for pt in patch_targets_abs]
                self._update_metrics(outputs, raw_patch_targets_for_map, device, torch.tensor([self.patch_size, self.patch_size], device=device))

        avg_loss = sum(losses) / len(losses)
        self.log("val/loss", avg_loss, on_epoch=True, on_step=False, prog_bar=True, batch_size=len(images))
        return avg_loss

    def on_validation_epoch_end(self):
        metrics = self.val_metric.compute()
        self.log("val/mAP", metrics["map"], prog_bar=True)
        self.log("val/mAP_50", metrics["map_50"], prog_bar=True)
        self.val_metric.reset()

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def on_test_epoch_end(self):
        return self.on_validation_epoch_end()

    def configure_optimizers(self):
        lora_params, dino_params = [], []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if "foundation_backbone" in name or "backbone.0.foundation_backbone" in name:
                lora_params.append(param)
            else:
                dino_params.append(param)

        optimizer = torch.optim.AdamW(
            [
                {"params": lora_params, "lr": self.lr_backbone},
                {"params": dino_params, "lr": self.lr},
            ],
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs, eta_min=1e-7
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}

    @staticmethod
    def collate_fn(batch):
        images = torch.stack([b[0] for b in batch]).float()
        targets = [b[1] for b in batch]
        return images, targets
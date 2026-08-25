
from __future__ import annotations
from transformers import SamModel
import timm
from transformers import CLIPVisionConfig, CLIPVisionModel
from transformers import AutoModel
import time
import math
from abc import ABC, abstractmethod
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class AllMLPDecoder(nn.Module):
    """SegFormer-style all-MLP head for uniform-resolution ViT feature levels."""
    def __init__(self, in_channels_list, embed_dim=256, num_classes=2):
        super().__init__()
        self.projs = nn.ModuleList(
            [nn.Conv2d(c, embed_dim, 1, bias=False) for c in in_channels_list]
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(embed_dim * len(in_channels_list), embed_dim, 1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Conv2d(embed_dim, num_classes, 1)

    def forward(self, feature_maps):
        # All levels are same spatial size for ViT (unlike MiT which has pyramid)
        projected = [proj(f) for proj, f in zip(self.projs, feature_maps)]
        # Upsample all to the first (finest) level — here all are equal
        target = projected[0].shape[-2:]
        aligned = [
            F.interpolate(p, size=target, mode="bilinear", align_corners=False)
            for p in projected
        ]
        x = self.fuse(torch.cat(aligned, dim=1))
        logits = self.head(x)
        return F.interpolate(logits, scale_factor=16, mode="bilinear", align_corners=False)



class ResidualConvUnit(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
        )

    def forward(self, x):
        return x + self.conv(x)


class DPTFusionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.rcu1 = ResidualConvUnit(channels)
        self.rcu2 = ResidualConvUnit(channels)
        self.project = nn.Conv2d(channels, channels, 1, bias=False)

    def forward(self, x, skip=None):
        if skip is not None:
            # skip is at the same resolution as the pre-upsample x; upsample skip to match x
            x = x + F.interpolate(self.rcu1(skip), size=x.shape[-2:], mode="bilinear", align_corners=False)
        x = self.rcu2(x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        return self.project(x)


class DPTDecoder(nn.Module):
    """Simplified DPT decoder: 4-level reassemble + top-down fusion blocks."""
    def __init__(self, in_channels=768, decoder_channels=128, num_classes=2):
        super().__init__()
        dc = decoder_channels
        # Use a list comprehension — [Conv2d(...)] * 4 would share a single instance
        self.projs = nn.ModuleList([nn.Conv2d(in_channels, dc, 1, bias=False) for _ in range(4)])
        # 4 fusion blocks (coarse→fine, each 2× upsample): 16× total from stride-16 → stride-1
        self.fusion3 = DPTFusionBlock(dc)  # stride 16 → 8
        self.fusion2 = DPTFusionBlock(dc)  # stride 8  → 4
        self.fusion1 = DPTFusionBlock(dc)  # stride 4  → 2
        self.fusion0 = DPTFusionBlock(dc)  # stride 2  → 1 (full input resolution)
        self.head = nn.Conv2d(dc, num_classes, 1)

    def forward(self, feature_maps):
        # feature_maps: coarse[0]…fine[3], all same spatial size (ViT uniform stride)
        l0, l1, l2, l3 = [proj(f) for proj, f in zip(self.projs, feature_maps)]
        # top-down fusion — each block upsamples x by 2×, then skip is aligned to that size
        x = self.fusion3(l3)
        x = self.fusion2(x, l2)
        x = self.fusion1(x, l1)
        x = self.fusion0(x, l0)
        logits = self.head(x)
        # x is now at stride-1 of the patch grid (i.e. H/16 * 2^4 = H)
        return logits  # already at input resolution




class PPM(nn.Module):
    def __init__(self, in_channels, out_channels, pool_scales=(1, 2, 3, 6)):
        super().__init__()
        self.pools = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(s),
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
            for s in pool_scales
        ])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels + out_channels * len(pool_scales), out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        H, W = x.shape[-2:]
        feats = [x] + [
            F.interpolate(pool(x), size=(H, W), mode="bilinear", align_corners=False)
            for pool in self.pools
        ]
        return self.bottleneck(torch.cat(feats, dim=1))


class UPerNetDecoder(nn.Module):
    """UPerNet: PPM on deepest level + FPN top-down fusion."""
    def __init__(self, in_channels_list, decoder_channels=128, num_classes=2):
        super().__init__()
        dc = decoder_channels
        # PPM on the last (most semantic) level
        self.ppm = PPM(in_channels_list[-1], dc)
        # FPN lateral projections
        self.laterals = nn.ModuleList(
            [nn.Conv2d(c, dc, 1, bias=False) for c in in_channels_list]
        )
        # FPN output refinement
        self.fpn_outs = nn.ModuleList(
            [nn.Sequential(
                nn.Conv2d(dc, dc, 3, padding=1, bias=False),
                nn.BatchNorm2d(dc),
                nn.ReLU(inplace=True),
            )
            for _ in in_channels_list]
        )
        # Final fusion head
        self.head = nn.Sequential(
            nn.Conv2d(dc * len(in_channels_list), dc, 3, padding=1, bias=False),
            nn.BatchNorm2d(dc),
            nn.ReLU(inplace=True),
            nn.Conv2d(dc, num_classes, 1),
        )

    def forward(self, feature_maps):
        # feature_maps: list of L levels, all same spatial size (ViT uniform stride)
        laterals = [proj(f) for proj, f in zip(self.laterals, feature_maps)]
        # Inject PPM context into deepest lateral
        laterals[-1] = laterals[-1] + self.ppm(feature_maps[-1])
        # Top-down FPN (all same spatial size → no upsample needed here)
        for i in range(len(laterals) - 2, -1, -1):
            laterals[i] = laterals[i] + F.interpolate(
                laterals[i + 1], size=laterals[i].shape[-2:],
                mode="bilinear", align_corners=False
            )
        fpn_outs = [refine(lat) for refine, lat in zip(self.fpn_outs, laterals)]
        # Upsample all to finest level and fuse
        target = fpn_outs[0].shape[-2:]
        fpn_outs = [
            F.interpolate(f, size=target, mode="bilinear", align_corners=False)
            for f in fpn_outs
        ]
        logits = self.head(torch.cat(fpn_outs, dim=1))
        return F.interpolate(logits, scale_factor=16, mode="bilinear", align_corners=False)



class Mask2FormerHead(nn.Module):
    """Simplified Mask2Former decoder: FPN pixel decoder + 1-layer masked-attention transformer."""
    def __init__(self, in_channels=768, decoder_channels=128, num_classes=2, num_queries=10, num_layers=3):
        super().__init__()
        dc = decoder_channels
        # Lightweight pixel decoder (FPN)
        self.pixel_proj = nn.Conv2d(in_channels, dc, 1, bias=False)
        self.pixel_refine = nn.Sequential(
            nn.Conv2d(dc, dc, 3, padding=1, bias=False),
            nn.GroupNorm(8, dc),
            nn.ReLU(inplace=True),
        )
        # Learnable queries
        self.queries = nn.Embedding(num_queries, dc)
        self.query_pe = nn.Embedding(num_queries, dc)  # learned positional encoding
        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=dc, nhead=8, dim_feedforward=dc * 4,
            dropout=0.0, batch_first=True, norm_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        # Heads
        self.class_head = nn.Linear(dc, num_classes)
        self.mask_head = nn.Sequential(
            nn.Linear(dc, dc), nn.ReLU(inplace=True), nn.Linear(dc, dc)
        )

    def forward(self, feature_maps):
        B = feature_maps[0].shape[0]
        # Pixel decoder: average the 4 levels then refine
        pixel_feat = sum(self.pixel_proj(f) for f in feature_maps) / len(feature_maps)
        pixel_feat = self.pixel_refine(pixel_feat)  # [B, dc, H/16, W/16]
        H, W = pixel_feat.shape[-2:]

        # Flatten pixel features for cross-attention memory
        memory = pixel_feat.flatten(2).permute(0, 2, 1)  # [B, H*W, dc]

        # Queries
        q = (self.queries.weight + self.query_pe.weight).unsqueeze(0).expand(B, -1, -1)  # [B, Q, dc]

        # Transformer decoder (simplified: no mask attention bias for clarity)
        q = self.transformer_decoder(q, memory)  # [B, Q, dc]

        # Per-query class and mask predictions
        class_logits = self.class_head(q)          # [B, Q, num_classes]
        mask_emb = self.mask_head(q)               # [B, Q, dc]

        # Mask logits via dot product with pixel features
        mask_logits = torch.einsum("bqc,bchw->bqhw", mask_emb, pixel_feat)  # [B, Q, H/16, W/16]

        # Combine: weighted sum of per-query masks by class softmax score
        scores = class_logits.softmax(dim=-1)  # [B, Q, num_classes]
        # For each class, sum query masks weighted by that class score
        logits = torch.einsum("bqk,bqhw->bkhw", scores, mask_logits)  # [B, num_classes, H/16, W/16]
        return F.interpolate(logits, scale_factor=16, mode="bilinear", align_corners=False)




# ---------------------------------------------------------------------------
# Decoder (TPA-SAD, from segdino_v2)
# ---------------------------------------------------------------------------
def drop_path(x: torch.Tensor, drop_prob: float, training: bool) -> torch.Tensor:
    """Per-sample stochastic depth: zero the whole tensor for some batch items.

    Operates on the residual branch only; survivors are scaled by
    ``1 / (1 - drop_prob)`` so the expected value is unchanged.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep = 1.0 - drop_prob
    # broadcast mask over all non-batch dims
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    mask = x.new_empty(shape).bernoulli_(keep)
    return x * mask / keep


class ResidualDepthwiseBlock(nn.Module):
    def __init__(self, channels, use_group_norm=True, drop_prob=0.0, dropout=0.0):
        super().__init__()
        self.depthwise = nn.Conv2d(
            channels, channels, 3, padding=1, groups=channels, bias=False)
        self.pointwise = nn.Conv2d(channels, channels, 1, bias=False)
        self.norm = (
            nn.GroupNorm(min(32, channels), channels)
            if use_group_norm
            else nn.BatchNorm2d(channels)
        )
        self.act = nn.GELU()
        # Channel (spatial) dropout on the residual features; no-op when 0.
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.drop_prob = drop_prob
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        residual = self.act(self.norm(self.pointwise(self.depthwise(x))))
        residual = self.dropout(residual)
        return x + drop_path(self.gamma * residual, self.drop_prob, self.training)


class TPAResampleProject(nn.Module):
    def __init__(self, channels, scale_factor):
        super().__init__()
        self.scale_factor = scale_factor
        self.conv = nn.Conv2d(channels, channels, 3, padding=1, bias=False)

    def forward(self, x):
        if self.scale_factor != 1:
            x = F.interpolate(
                x,
                scale_factor=self.scale_factor,
                mode="bilinear",
                align_corners=False,
            )
        return self.conv(x)


class TPASADDecoder(nn.Module):
    def __init__(self, in_dims, decoder_channels=128, num_classes=2,
                 use_group_norm=True, drop_path_rate=0.0, dropout=0.0):
        super().__init__()
        assert len(in_dims) == 4

        # Linearly ramp drop-path across the 8 SAD blocks (intra 1-4, inter 4-1).
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, 8)]
        def rb(i): return ResidualDepthwiseBlock(
            decoder_channels, use_group_norm=use_group_norm, drop_prob=dpr[i])

        # TPA: project backbone features into decoder channels and align them to
        # the four spatial branches used by the decoder.
        self.token_projections = nn.ModuleList(
            [nn.Conv2d(channels, decoder_channels, 1, bias=False)
             for channels in in_dims]
        )
        self.tpa_branch_1 = TPAResampleProject(
            decoder_channels, scale_factor=8)
        self.tpa_branch_2 = TPAResampleProject(
            decoder_channels, scale_factor=4)
        self.tpa_branch_3 = TPAResampleProject(
            decoder_channels, scale_factor=2)
        self.tpa_branch_4 = TPAResampleProject(
            decoder_channels, scale_factor=1)

        # SAD: refine each branch independently, then fuse them from coarse to fine.
        self.sad_intra_1 = rb(0)
        self.sad_intra_2 = rb(1)
        self.sad_intra_3 = rb(2)
        self.sad_intra_4 = rb(3)

        self.sad_inter_4 = rb(4)
        self.sad_inter_3 = rb(5)
        self.sad_inter_2 = rb(6)
        self.sad_inter_1 = rb(7)

        # needs num_groups <= num_channels, hence the max(1, ...).
        # refine_channels = max(1, decoder_channels // 2)
        # self.refine_block = ResidualDepthwiseBlock(
        #     decoder_channels, use_group_norm=use_group_norm
        # )
        # self.refine_up = nn.ConvTranspose2d(
        #     decoder_channels, refine_channels, kernel_size=2, stride=2
        # )

        self.head_dropout = nn.Dropout2d(
            dropout) if dropout > 0 else nn.Identity()
        self.out_conv = nn.Conv2d(decoder_channels, num_classes, 1)

    def forward(self, feature_maps):
        # feature_maps: list of four [B, C, ph, pw] tensors (coarse-to-fine input).
        # TPA: project each level, then resample into the four-branch pyramid.
        branches = [proj(fmap) for proj, fmap in zip(
            self.token_projections, feature_maps)]

        branch_1 = self.tpa_branch_1(branches[0])
        branch_2 = self.tpa_branch_2(branches[1])
        branch_3 = self.tpa_branch_3(branches[2])
        branch_4 = self.tpa_branch_4(branches[3])

        # SAD: each branch is refined, then merged top-down.
        level_1 = self.sad_intra_1(branch_1)
        level_2 = self.sad_intra_2(branch_2)
        level_3 = self.sad_intra_3(branch_3)
        level_4 = self.sad_intra_4(branch_4)

        x4 = self.sad_inter_4(level_4)
        x3_up = F.interpolate(
            x4, size=level_3.shape[-2:], mode="bilinear", align_corners=False)
        x3 = self.sad_inter_3(x3_up + level_3)

        x2_up = F.interpolate(
            x3, size=level_2.shape[-2:], mode="bilinear", align_corners=False)
        x2 = self.sad_inter_2(x2_up + level_2)

        x1_up = F.interpolate(
            x2, size=level_1.shape[-2:], mode="bilinear", align_corners=False)
        x1 = self.sad_inter_1(x1_up + level_1)

        logits = self.out_conv(self.head_dropout(x1))
        return F.interpolate(logits, scale_factor=2, mode="bilinear", align_corners=False)


# ── Building blocks ───────────────────────────────────────────────────────────

class SEGate(nn.Module):
    """Squeeze-and-Excitation channel gate: global avg pool → fc → sigmoid → scale."""
    def __init__(self, channels, reduction=4):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.gate(x).view(x.shape[0], -1, 1, 1)


class SEDepthwiseBlock(nn.Module):
    """Depthwise residual block with SE channel recalibration."""
    def __init__(self, channels):
        super().__init__()
        norm = nn.GroupNorm(max(1, channels // 16), channels)
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            norm,
            nn.GELU(),
            nn.Conv2d(channels, channels, 1, bias=False),
        )
        self.se = SEGate(channels)

    def forward(self, x):
        return x + self.se(self.block(x))


class LearnedFusionGate(nn.Module):
    """Per-channel learned gate: output = sigmoid(α) * top_down + (1 - sigmoid(α)) * skip."""
    def __init__(self, channels):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(1, channels, 1, 1))  # init → 0.5 blend

    def forward(self, top_down, skip):
        a = torch.sigmoid(self.alpha)
        return a * top_down + (1.0 - a) * skip


class LightPPM(nn.Module):
    """Lightweight PPM: 3 pool scales at dc//4 channels, fused back to dc."""
    def __init__(self, channels, pool_scales=(1, 3, 6)):
        super().__init__()
        mid = max(channels // 4, 8)
        self.pools = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(s),
                nn.Conv2d(channels, mid, 1, bias=False),
                nn.GroupNorm(1, mid),
                nn.GELU(),
            )
            for s in pool_scales
        ])
        self.fuse = nn.Sequential(
            nn.Conv2d(channels + mid * len(pool_scales), channels, 1, bias=False),
            nn.GroupNorm(max(1, channels // 16), channels),
            nn.GELU(),
        )

    def forward(self, x):
        H, W = x.shape[-2:]
        pooled = [
            F.interpolate(pool(x), size=(H, W), mode="bilinear", align_corners=False)
            for pool in self.pools
        ]
        return self.fuse(torch.cat([x] + pooled, dim=1))


class TPAResampleProjectLN(nn.Module):
    """TPA resampling branch with LayerNorm — matches TPA-SAD's full 3×3 conv capacity
    but replaces GroupNorm with LayerNorm to align with ViT's internal normalisation."""
    def __init__(self, channels, scale_factor):
        super().__init__()
        self.scale_factor = scale_factor
        # Full 3×3 conv (same budget as TPA-SAD's TPAResampleProject)
        self.conv = nn.Conv2d(channels, channels, 3, padding=1, bias=False)

    def forward(self, x):
        if self.scale_factor != 1:
            x = F.interpolate(
                x,
                size=(x.shape[-2] * self.scale_factor, x.shape[-1] * self.scale_factor),
                mode="bilinear", align_corners=False,
            )
        # LayerNorm over channel dim (matches ViT's per-token norm, unlike TPA-SAD's GroupNorm)
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, [x.shape[-1]])
        x = x.permute(0, 3, 1, 2)
        return F.gelu(self.conv(x))


# ── PyraFuse decoder ──────────────────────────────────────────────────────────

class PyraFuseDecoder(nn.Module):
    """
    PyraFuse: improved TPA pyramid decoder at the same ~1.5M param budget as TPA-SAD.

    Improvements over SegDINO TPA-SAD:
      • LayerNorm in TPA branches (matches ViT's internal norm)
      • SE-gated depthwise residual blocks (channel recalibration)
      • Learned per-channel α-gate for inter-scale fusion (replaces blind add)
      • LightPPM global context on coarsest branch
      • 3×3 dw+pw boundary refinement before head
    """
    def __init__(self, in_dims, decoder_channels=128, num_classes=2):
        super().__init__()
        assert len(in_dims) == 4
        dc = decoder_channels

        self.token_projs = nn.ModuleList(
            [nn.Conv2d(c, dc, 1, bias=False) for c in in_dims]
        )

        self.tpa_branch_1 = TPAResampleProjectLN(dc, scale_factor=8)
        self.tpa_branch_2 = TPAResampleProjectLN(dc, scale_factor=4)
        self.tpa_branch_3 = TPAResampleProjectLN(dc, scale_factor=2)
        self.tpa_branch_4 = TPAResampleProjectLN(dc, scale_factor=1)

        self.ppm = LightPPM(dc)   # global context on coarsest branch

        self.intra_1 = SEDepthwiseBlock(dc)
        self.intra_2 = SEDepthwiseBlock(dc)
        self.intra_3 = SEDepthwiseBlock(dc)
        self.intra_4 = SEDepthwiseBlock(dc)

        self.gate_4to3 = LearnedFusionGate(dc)
        self.gate_3to2 = LearnedFusionGate(dc)
        self.gate_2to1 = LearnedFusionGate(dc)

        self.inter_4 = SEDepthwiseBlock(dc)
        self.inter_3 = SEDepthwiseBlock(dc)
        self.inter_2 = SEDepthwiseBlock(dc)
        self.inter_1 = SEDepthwiseBlock(dc)

        self.head_refine = nn.Sequential(
            nn.Conv2d(dc, dc, 3, padding=1, groups=dc, bias=False),
            nn.Conv2d(dc, dc, 1, bias=False),
            nn.GroupNorm(max(1, dc // 16), dc),
            nn.GELU(),
        )
        self.out_conv = nn.Conv2d(dc, num_classes, 1)

    def forward(self, feature_maps):
        projs = [proj(f) for proj, f in zip(self.token_projs, feature_maps)]

        b1 = self.tpa_branch_1(projs[0])
        b2 = self.tpa_branch_2(projs[1])
        b3 = self.tpa_branch_3(projs[2])
        b4 = self.tpa_branch_4(projs[3])

        b4 = b4 + self.ppm(b4)

        l1 = self.intra_1(b1)
        l2 = self.intra_2(b2)
        l3 = self.intra_3(b3)
        l4 = self.intra_4(b4)

        x4 = self.inter_4(l4)
        x3 = self.inter_3(self.gate_4to3(
            F.interpolate(x4, size=l3.shape[-2:], mode="bilinear", align_corners=False), l3))
        x2 = self.inter_2(self.gate_3to2(
            F.interpolate(x3, size=l2.shape[-2:], mode="bilinear", align_corners=False), l2))
        x1 = self.inter_1(self.gate_2to1(
            F.interpolate(x2, size=l1.shape[-2:], mode="bilinear", align_corners=False), l1))

        logits = self.out_conv(self.head_refine(x1))
        return F.interpolate(logits, scale_factor=2, mode="bilinear", align_corners=False)

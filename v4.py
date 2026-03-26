import torch
import torch.nn as nn
import torch.nn.functional as F
# ---------------------------------------------------------------------------
# Pyramid Pooling Module  (reusable at any scale)
# ---------------------------------------------------------------------------
class PyramidPoolingModule(nn.Module):
    def __init__(self, in_dim, out_dim, pool_scales=(1, 2, 3, 6)):
        super().__init__()
 
        self.pre = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
        )
 
        branch_dim = out_dim // len(pool_scales)
        self.stages = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(scale),
                nn.Conv2d(out_dim, branch_dim, 1),
                nn.BatchNorm2d(branch_dim),
                nn.ReLU(inplace=True),
            )
            for scale in pool_scales
        ])
 
        total_in = out_dim + branch_dim * len(pool_scales)
        self.fuse = nn.Sequential(
            nn.Conv2d(total_in, out_dim, 3, padding=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
        )
 
    def forward(self, x):
        x    = self.pre(x)
        size = x.shape[-2:]
 
        feats = [x]
        for stage in self.stages:
            up = F.interpolate(stage(x), size=size, mode='bilinear', align_corners=False)
            feats.append(up)
 
        return self.fuse(torch.cat(feats, dim=1))
 
# ---------------------------------------------------------------------------
# Cross-Attention Gate
# --------------------------------------------------------------------------- 
class CrossAttentionGate(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.reduction = reduction
        inner = max(channels // 8, 16)

        self.q_proj = nn.Conv2d(channels, inner, 1)
        self.k_proj = nn.Conv2d(channels, inner, 1)
        self.v_proj = nn.Conv2d(channels, channels, 1)

        self.gate_out = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid(),
        )

        self.scale = inner ** -0.5

    def forward(self, F_high: torch.Tensor, F_low: torch.Tensor) -> torch.Tensor:
        B, C, H, W = F_high.shape
        stride = self.reduction

        # ── Downsample BOTH Q and K/V to the same reduced grid ──────────
        # KEY CHANGE: V is also computed on the downsampled grid,
        # so attn @ V is a valid operation (dimensions align).
        # We upsample the *output*, not the attention weights.
        if stride > 1:
            q_feat = F.avg_pool2d(F_high, stride, stride)  # (B, C, Hs, Ws)
            kv_feat = F.avg_pool2d(F_low,  stride, stride)  # (B, C, Hs, Ws)
        else:
            q_feat  = F_high
            kv_feat = F_low

        Hs, Ws = q_feat.shape[-2:]
        N = Hs * Ws

        # ── Project to Q, K, V — all on the downsampled grid ────────────
        Q = self.q_proj(q_feat).reshape(B, -1, N).permute(0, 2, 1)   # (B, N, inner)
        K = self.k_proj(kv_feat).reshape(B, -1, N).permute(0, 2, 1)  # (B, N, inner)
        V = self.v_proj(kv_feat).reshape(B, C, N).permute(0, 2, 1)   # (B, N, C)

        # ── Scaled dot-product attention ─────────────────────────────────
        attn = torch.softmax(
            torch.bmm(Q, K.transpose(1, 2)) * self.scale, dim=-1
        )  # (B, N, N)

        # ── THE FIX: proper weighted sum — each query gets its own output ─
        # out[i] = sum_j( attn[i,j] * V[j] )  ← spatially selective
        attended = torch.bmm(attn, V)           # (B, N, C)  ← not mean(), bmm()
        attended = attended.permute(0, 2, 1).reshape(B, C, Hs, Ws)  # (B, C, Hs, Ws)

        # ── Upsample the *attended output* back to full resolution ───────
        # We upsample features, not attention weights — much cleaner.
        attended = F.interpolate(attended, size=(H, W),
                                 mode='bilinear', align_corners=False)  # (B, C, H, W)

        # Gate derived from the properly attended features
        gate = self.gate_out(attended)          # (B, C, H, W) ∈ (0, 1)
        return gate 
 
# ---------------------------------------------------------------------------
# Gated Fusion  (cross-attention augmented)
# ---------------------------------------------------------------------------
class GatedFusion(nn.Module):
    def __init__(self, in_high: int, in_low: int, out_channels: int,
                 attn_reduction: int = 4):
        super().__init__()
 
        self.high_proj = nn.Conv2d(in_high, out_channels, 1)
        self.low_proj  = nn.Conv2d(in_low,  out_channels, 1)
 
        # Cross-attention gate replaces the plain gate_conv
        self.ca_gate = CrossAttentionGate(out_channels, reduction=attn_reduction)
 
        # Optional residual channel-attention (SE-style) on top
        squeeze = max(out_channels // 16, 4)
        self.channel_se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, squeeze, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(squeeze, out_channels, 1),
            nn.Sigmoid(),
        )
 
        self.refine = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
 
    def forward(self, F_high: torch.Tensor, F_low: torch.Tensor) -> torch.Tensor:
        F_high = self.high_proj(F_high)
        F_low  = self.low_proj(F_low)
 
        # Align spatial resolution
        F_high = F.interpolate(F_high, size=F_low.shape[-2:],
                               mode='bilinear', align_corners=False)
 
        # Cross-attention gate: spatially selective, driven by high↔low interplay
        gate = self.ca_gate(F_high, F_low)          # (B, C, H, W)
 
        fused = gate * F_low + (1.0 - gate) * F_high
 
        # Channel-wise SE recalibration
        fused = fused * self.channel_se(fused)
 
        return self.refine(fused)
 
# ---------------------------------------------------------------------------
# Upsample + refine block
# ---------------------------------------------------------------------------
class UpsampleRefine(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.reduce = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.refine = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
 
    def forward(self, x: torch.Tensor, target_size) -> torch.Tensor:
        x = self.reduce(x)
        x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
        return self.refine(x)
 
# ---------------------------------------------------------------------------
# Decoder Stage  (PPM injected at every scale)
# --------------------------------------------------------------------------- 
class DecoderStage(nn.Module):
    def __init__(self, in_high: int, in_low: int, out_channels: int,
                 use_ppm: bool = True,
                 ppm_scales=(1, 2, 3, 6),
                 attn_reduction: int = 4):
        super().__init__()
 
        self.ppm = (
            PyramidPoolingModule(in_low, in_low, pool_scales=ppm_scales)
            if use_ppm else None
        )
 
        self.upsample = UpsampleRefine(in_high, out_channels)
        self.fusion   = GatedFusion(out_channels, in_low, out_channels,
                                    attn_reduction=attn_reduction)
 
    def forward(self, F_high: torch.Tensor, F_low: torch.Tensor) -> torch.Tensor:
        if self.ppm is not None:
            F_low = self.ppm(F_low)   # enrich skip with multi-scale context
 
        F_high_up = self.upsample(F_high, F_low.shape[-2:])
        return self.fusion(F_high_up, F_low)
 
 
# ---------------------------------------------------------------------------
# Full Segmentation Decoder
# ---------------------------------------------------------------------------
class SegmentationDecoder(nn.Module):
    _PPM_SCALES = {
        "stage3": (1, 2, 3, 6),   # deepest skip (C2) — full global context
        "stage2": (1, 2, 3),       # mid-level   (C1)
        "stage1": (1, 2),          # finest skip (C0) — minimal pooling
    }
 
    _ATTN_REDUCTION = {
        "stage3": 2,
        "stage2": 4,
        "stage1": 8,
    }
 
    def __init__(self, dims, num_classes: int, decoder_dim: int = 256):
        super().__init__()
        self.decoder_dim = decoder_dim
 
        self.ppm_deep = PyramidPoolingModule(dims[3], decoder_dim,
                                             pool_scales=(1, 2, 3, 6))
 
        self.stage3 = DecoderStage(
            decoder_dim, dims[2], decoder_dim,
            use_ppm=True,
            ppm_scales=self._PPM_SCALES["stage3"],
            attn_reduction=self._ATTN_REDUCTION["stage3"],
        )
        self.stage2 = DecoderStage(
            decoder_dim, dims[1], decoder_dim,
            use_ppm=True,
            ppm_scales=self._PPM_SCALES["stage2"],
            attn_reduction=self._ATTN_REDUCTION["stage2"],
        )
        self.stage1 = DecoderStage(
            decoder_dim, dims[0], decoder_dim,
            use_ppm=True,
            ppm_scales=self._PPM_SCALES["stage1"],
            attn_reduction=self._ATTN_REDUCTION["stage1"],
        )
 
        self.final_refine = nn.Sequential(
            nn.Conv2d(decoder_dim, decoder_dim, 3, padding=1),
            nn.BatchNorm2d(decoder_dim),
            nn.ReLU(inplace=True),
        )
 
        self.classifier = nn.Conv2d(decoder_dim, num_classes, 1)
 
    def forward(self, features, input_size):
        F0, F1, F2, F3 = features
 
        F3 = self.ppm_deep(F3)
 
        D2 = self.stage3(F3, F2)
        D1 = self.stage2(D2, F1)
        D0 = self.stage1(D1, F0)
 
        out = self.final_refine(D0)
        out = F.interpolate(out, size=input_size, mode='bilinear', align_corners=False)
        return self.classifier(out)
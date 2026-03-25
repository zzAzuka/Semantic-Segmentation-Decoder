import torch
import torch.nn as nn
import torch.nn.functional as F

class UpsampleRefine(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.reduce = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        self.refine = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, target_size):
        x = self.reduce(x)
        x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
        x = self.refine(x)
        return x

class GatedFusion(nn.Module):
    def __init__(self, in_high, in_low, out_channels):
        super().__init__()

        self.high_proj = nn.Conv2d(in_high, out_channels, 1)
        self.low_proj  = nn.Conv2d(in_low,  out_channels, 1)

        self.gate_conv = nn.Conv2d(out_channels * 2, out_channels, 1)

        self.refine = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, F_high, F_low):
        F_high = self.high_proj(F_high)
        F_low  = self.low_proj(F_low)

        F_high = F.interpolate(F_high, size=F_low.shape[-2:], mode='bilinear', align_corners=False)

        cat = torch.cat([F_high, F_low], dim=1)

        gate = torch.sigmoid(self.gate_conv(cat))

        # Stabilized fusion (prevents collapse)
        Fusion = gate * F_low + (1 - gate) * F_high

        return self.refine(Fusion)

class DecoderStage(nn.Module):
    def __init__(self, in_high, in_low, out_channels):
        super().__init__()

        self.upsample = UpsampleRefine(in_high, out_channels)
        self.fusion   = GatedFusion(out_channels, in_low, out_channels)

    def forward(self, F_high, F_low):
        F_high_up = self.upsample(F_high, F_low.shape[-2:])
        return self.fusion(F_high_up, F_low)


class PyramidPoolingModule(nn.Module):
    def __init__(self, in_dim, out_dim, pool_scales=(1, 2, 3, 6)):
        super().__init__()

        self.pre = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True)
        )

        self.stages = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(scale),
                nn.Conv2d(out_dim, out_dim // len(pool_scales), 1),
                nn.BatchNorm2d(out_dim // len(pool_scales)),
                nn.ReLU(inplace=True)
            )
            for scale in pool_scales
        ])

        # Correct channel math — works for any pool_scales length
        branch_dim = out_dim // len(pool_scales)
        total_in   = out_dim + branch_dim * len(pool_scales)
        self.fuse  = nn.Sequential(                          # BN + ReLU restored
            nn.Conv2d(total_in, out_dim, 3, padding=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x    = self.pre(x)
        size = x.shape[-2:]

        feats = [x]
        for stage in self.stages:
            pooled = stage(x)
            up     = F.interpolate(pooled, size=size, mode='bilinear', align_corners=False)
            feats.append(up)

        x = torch.cat(feats, dim=1)
        return self.fuse(x)

class SegmentationDecoder(nn.Module):
    def __init__(self, dims, num_classes, decoder_dim=256):
        """
        dims: list of backbone channels [F0, F1, F2, F3]
        """
        super().__init__()

        self.decoder_dim = decoder_dim

        # Global context on deepest feature
        self.ppm = PyramidPoolingModule(dims[3], decoder_dim)

        # Decoder stages
        self.stage3 = DecoderStage(decoder_dim, dims[2], decoder_dim)
        self.stage2 = DecoderStage(decoder_dim, dims[1], decoder_dim)
        self.stage1 = DecoderStage(decoder_dim, dims[0], decoder_dim)

        self.final_refine = nn.Sequential(
            nn.Conv2d(decoder_dim, decoder_dim, 3, padding=1),
            nn.BatchNorm2d(decoder_dim),
            nn.ReLU(inplace=True)
        )

        self.classifier = nn.Conv2d(decoder_dim, num_classes, 1)

    def forward(self, features, input_size):
        F0, F1, F2, F3 = features

        # Inject global context
        F3 = self.ppm(F3)

        D2 = self.stage3(F3, F2)
        D1 = self.stage2(D2, F1)
        D0 = self.stage1(D1, F0)

        out = self.final_refine(D0)

        out = F.interpolate(out, size=input_size, mode='bilinear', align_corners=False)

        return self.classifier(out)
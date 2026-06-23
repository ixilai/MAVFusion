# MAVFusion: MDIM (Motion-Guided Dual-Interaction Module) and its building blocks
# Static branch (StaticUNetBlock, LightEncoder) handles weak interaction in static regions;
# dynamic branch (MultiScaleSparseBranch, MaskedDATBlock) performs Top-K sparse attention
# in dynamic regions identified by the motion mask.

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ---------------------------------------------------------------------------
# Static branch: StaticUNetBlock (lightweight U-Net for local texture modeling)
# ---------------------------------------------------------------------------
class StaticUNetBlock(nn.Module):
    def __init__(self, dim, expansion_ratio=2, bias=False):
        super().__init__()
        exp_dim = int(dim * expansion_ratio)

        # Encoder
        self.enc_conv1 = nn.Conv2d(dim, exp_dim, kernel_size=3, padding=1, bias=bias)
        self.enc_dw = nn.Conv2d(exp_dim, exp_dim, kernel_size=3, padding=1, groups=exp_dim, bias=bias)
        self.enc_norm = nn.GroupNorm(4, exp_dim)

        # Bottleneck (Scale-Space)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(exp_dim, exp_dim, kernel_size=5, padding=2, groups=exp_dim, bias=bias),
            nn.GELU(),
            nn.Conv2d(exp_dim, exp_dim, kernel_size=3, padding=1, groups=exp_dim, bias=bias)
        )

        # Decoder
        self.dec_pw = nn.Conv2d(exp_dim, exp_dim, kernel_size=1, bias=bias)
        self.dec_conv2 = nn.Conv2d(exp_dim, dim, kernel_size=3, padding=1, bias=bias)

        self.act = nn.GELU()
        self.gamma = nn.Parameter(torch.ones(1, dim, 1, 1) * 0.1)

    def forward(self, x):
        identity = x
        e1 = self.act(self.enc_norm(self.enc_dw(self.enc_conv1(x))))
        low_res = F.avg_pool2d(e1, kernel_size=2, stride=2)
        b = self.bottleneck(low_res)
        up = F.interpolate(b, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False)
        fusion = self.act(self.dec_pw(e1 + up))
        out = self.dec_conv2(fusion)
        return identity + self.gamma * out


class MaskedDATBlock(nn.Module):
    def __init__(self, dim, num_heads=4, topk=32, ffn_expansion=2.66, bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.topk = topk
        self.scale = (dim // num_heads) ** -0.5

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=bias)
        self.proj = nn.Linear(dim, dim)

        hidden_dim = int(dim * ffn_expansion)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden_dim, bias=bias),
            nn.GELU(),
            nn.Linear(hidden_dim, dim, bias=bias)
        )

    def forward(self, x, global_tokens):
        Bp, C, p1, p2 = x.shape
        x_token = x.flatten(2).mean(-1)
        residual = x_token
        x_norm = self.norm1(x_token)

        # Joint QKV: query from active patch, key/value from global tokens
        qkv = self.qkv(torch.cat([x_norm, global_tokens], dim=0))
        q, k, v = qkv.chunk(3, dim=-1)

        q = q[:Bp]
        k, v = k[Bp:], v[Bp:]

        q = rearrange(q, 'n (h d) -> h n d', h=self.num_heads)
        k = rearrange(k, 'm (h d) -> h m d', h=self.num_heads)
        v = rearrange(v, 'm (h d) -> h m d', h=self.num_heads)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        tk = min(self.topk, attn.shape[-1])
        val, idx = attn.topk(tk, dim=-1)
        attn_sparse = torch.zeros_like(attn).scatter_(-1, idx, val).softmax(dim=-1)

        out = rearrange(attn_sparse @ v, 'h n d -> n (h d)')
        x_token = residual + self.proj(out)
        x_token = x_token + self.ffn(self.norm2(x_token))
        return x_token.view(Bp, C, 1, 1).expand(-1, -1, p1, p2)


class MultiScaleSparseBranch(nn.Module):
    def __init__(self, dim, num_blocks, head=4, patch_size=8, tau=0.3, topk_ratio=0.3):
        super().__init__()
        self.patch_size = patch_size
        self.tau, self.topk_ratio = tau, topk_ratio

        self.compress = nn.Conv2d(dim, dim, 1, bias=False)
        self.excavate = nn.Conv2d(dim, dim, 1, bias=False)

        self.blocks = nn.ModuleList([
            MaskedDATBlock(dim, num_heads=head, topk=64)
            for _ in range(num_blocks)
        ])

        self.proj_out = nn.Conv2d(dim, dim, 1)

    def forward(self, x, mask):
        B, C, H, W = x.shape
        p = self.patch_size
        if isinstance(p, (list, tuple)):
            p = p[0]
        x_comp = self.compress(x)

        # Gradient sink prevents DDP "unused parameters" warnings by
        # forcing a 0-coefficient connection to all block parameters.
        dummy_grad_sink = sum(p.view(-1)[0] for p in self.blocks.parameters()) * 0.0

        # Pad to a multiple of patch size
        if H < p or W < p:
            return torch.zeros_like(x_comp) + dummy_grad_sink

        hp, wp = ((H + p - 1) // p) * p, ((W + p - 1) // p) * p
        m_pad = F.pad(mask, (0, wp - W, 0, hp - H))

        # Per-patch saliency from motion mask
        scores = F.adaptive_avg_pool2d(m_pad, (hp // p, wp // p)).view(-1)
        k = max(1, int(self.topk_ratio * scores.numel()))
        val, active_idx = torch.topk(scores, k, sorted=False)

        # Split into patches; global context = mean over spatial dims
        x_pad = F.pad(x_comp, (0, wp - W, 0, hp - H))
        x_p = rearrange(x_pad, 'b c (nh p1) (nw p2) -> (b nh nw) c p1 p2', p1=p, p2=p)
        g_tokens = x_p.mean(dim=(2, 3))

        # Apply attention only to active patches
        act_p = self.compress(x_p[active_idx])
        for blk in self.blocks:
            act_p = blk(act_p, g_tokens)

        # Scatter back with saliency-weighted contribution
        soft_weights = val.view(-1, 1, 1, 1)
        act_p = self.excavate(act_p) * soft_weights

        diff = torch.zeros_like(x_p)
        diff.index_copy_(0, active_idx.long(), act_p)

        diff_map = rearrange(diff, '(b nh nw) c p1 p2 -> b c (nh p1) (nw p2)',
                             b=B, nh=hp // p, nw=wp // p)[:, :, :H, :W]

        return self.proj_out(diff_map + dummy_grad_sink)


class HybridBlock(nn.Module):
    def __init__(self, dim, num_blocks=1, head=4, patch_sizes=(8, 16)):
        super().__init__()
        self.static_branch = StaticUNetBlock(dim)
        self.dynamic_branch = MultiScaleSparseBranch(dim, num_blocks, head, patch_sizes)
        self.smooth = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=False)

    def forward(self, x, mask):
        s_feat = self.static_branch(x)
        d_feat = self.dynamic_branch(x, mask)

        m_s = F.interpolate(mask, size=x.shape[-2:], mode='bilinear', align_corners=False)
        out = s_feat + d_feat * m_s
        return out + self.smooth(out)


class LightEncoder(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, 3, 1, 1, groups=in_dim),
            nn.Conv2d(in_dim, out_dim, 1),
            nn.GELU()
        )
        self.smooth = nn.Conv2d(out_dim, out_dim, 3, 1, 1, groups=out_dim)

    def forward(self, x):
        out = self.body(x)
        return out + self.smooth(out)


class Fusion_Net(nn.Module):
    def __init__(self, dim=64, num_blocks=1, head=4):
        super().__init__()

        # Input projection: 2 * (dim//8) -> dim//4
        self.en_ir = LightEncoder(dim, dim // 8)
        self.en_vi = LightEncoder(dim, dim // 8)
        curr = dim // 4

        # --- Encoder ---
        self.e1 = HybridBlock(curr, num_blocks, head)
        self.d1 = nn.Conv2d(curr, curr * 2, 3, 2, 1)

        self.e2 = HybridBlock(curr * 2, num_blocks, head)
        self.d2 = nn.Conv2d(curr * 2, curr * 4, 3, 2, 1)

        self.e3 = HybridBlock(curr * 4, num_blocks, head)
        self.d3 = nn.Conv2d(curr * 4, curr * 4, 3, 2, 1)

        # --- Bottleneck ---
        self.bt = HybridBlock(curr * 4, num_blocks, head)

        # --- Decoder (channel widths in comments denote 2x up-concat channel counts) ---
        self.up3 = nn.ConvTranspose2d(curr * 4, curr * 4, 2, 2)
        self.re3 = nn.Conv2d(curr * 8, curr * 4, 1)
        self.dec3 = HybridBlock(curr * 4, num_blocks, head)

        self.up2 = nn.ConvTranspose2d(curr * 4, curr * 2, 2, 2)
        self.re2 = nn.Conv2d(curr * 4, curr * 2, 1)
        self.dec2 = HybridBlock(curr * 2, num_blocks, head)

        self.up1 = nn.ConvTranspose2d(curr * 2, curr, 2, 2)
        self.re1 = nn.Conv2d(curr * 2, curr, 1)
        self.dec1 = HybridBlock(curr, num_blocks, head)

        # Reconstruction
        self.recon = nn.Sequential(
            nn.Conv2d(curr, dim, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(dim, 3, 3, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, ir, vi, mask):
        x = torch.cat([self.en_ir(ir), self.en_vi(vi)], dim=1)

        skip1 = self.e1(x, mask)
        x = self.d1(skip1)

        skip2 = self.e2(x, mask)
        x = self.d2(skip2)

        skip3 = self.e3(x, mask)
        x = self.d3(skip3)

        x = self.bt(x, mask)

        x = self.up3(x)
        x = self.dec3(self.re3(torch.cat([x, skip3], 1)), mask)

        x = self.up2(x)
        x = self.dec2(self.re2(torch.cat([x, skip2], 1)), mask)

        x = self.up1(x)
        x = self.dec1(self.re1(torch.cat([x, skip1], 1)), mask)

        return self.recon(x)
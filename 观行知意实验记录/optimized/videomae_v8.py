"""Domain adaptation for VideoMAE V2 using only paired depth and IR."""
import torch
from torch import nn
from torch.nn import functional as F
from videomae_model import VideoMAEClassifier


class DualPatchEmbed(nn.Module):
    def __init__(self, source):
        super().__init__()
        src = source.proj
        self.depth = nn.Conv3d(3, src.out_channels, src.kernel_size, src.stride)
        self.ir = nn.Conv3d(1, src.out_channels, src.kernel_size, src.stride, bias=False)
        self.depth_adapter = self._adapter(src.out_channels)
        self.ir_adapter = self._adapter(src.out_channels)
        with torch.no_grad():
            self.depth.weight.copy_(src.weight[:, :3])
            self.depth.bias.copy_(src.bias)
            self.ir.weight.copy_(src.weight[:, 3:])

    @staticmethod
    def _adapter(dim):
        block = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 32), nn.GELU(), nn.Linear(32, dim))
        nn.init.zeros_(block[-1].weight)
        nn.init.zeros_(block[-1].bias)
        return block

    def streams(self, x):
        depth = self.depth(x[:, :3]).flatten(2).transpose(1, 2)
        ir = self.ir(x[:, 3:]).flatten(2).transpose(1, 2)
        return depth + self.depth_adapter(depth), ir + self.ir_adapter(ir)

    def forward(self, x):
        depth, ir = self.streams(x)
        if hasattr(self, 'sensor_gate'):
            # Residual reliability correction starts at the exact equal-sum solution.
            a = F.layer_norm(depth, (depth.shape[-1],))
            b = F.layer_norm(ir, (ir.shape[-1],))
            gate = .5 * self.sensor_gate(torch.cat((a, b), dim=-1)).tanh()
            return depth + ir + gate * (depth - ir)
        return depth + ir


class QVLoRA(nn.Module):
    """Rank-8 residuals on attention query and value projections."""
    def __init__(self, attention, rank=8):
        super().__init__()
        self.base = attention
        dim = attention.qkv.in_features
        self.down = nn.Linear(dim, rank, bias=False)
        self.up = nn.Linear(rank, 2 * dim, bias=False)
        nn.init.zeros_(self.up.weight)
        self.scale = 2.0

    def forward(self, x):
        a = self.base
        batch, tokens, dim = x.shape
        bias = torch.cat((a.q_bias, torch.zeros_like(a.v_bias), a.v_bias))
        q, k, v = F.linear(x, a.qkv.weight, bias).chunk(3, dim=-1)
        dq, dv = (self.up(self.down(x)) * self.scale).chunk(2, dim=-1)
        qkv = [(t.reshape(batch, tokens, a.heads, dim // a.heads).transpose(1, 2))
               for t in (q + dq, k, v + dv)]
        value = F.scaled_dot_product_attention(*qkv)
        return a.proj(value.transpose(1, 2).reshape(batch, tokens, dim))


class TemporalAdapter(nn.Module):
    """Local motion correction on the 8×14×14 VideoMAE tubelet grid."""
    def __init__(self, dim, width=32):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.down = nn.Linear(dim, width)
        self.motion = nn.Conv3d(width, width, (3, 1, 1), padding=(1, 0, 0), groups=width)
        self.up = nn.Linear(width, dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, tokens):
        b, n, _ = tokens.shape
        feature = self.down(self.norm(tokens)).transpose(1, 2).reshape(b, -1, 8, 14, 14)
        shifted = self.motion(feature).flatten(2).transpose(1, 2).reshape(b, n, -1)
        return tokens + self.up(F.gelu(shifted))


class DomainVideoMAE(VideoMAEClassifier):
    def __init__(self, mode='dual', *, pretrained=None, warmstart=None, variant='plain'):
        super().__init__(pretrained=pretrained)
        if warmstart is not None:
            self.load_state_dict(warmstart, strict=True)
        self.encoder.patch_embed = DualPatchEmbed(self.encoder.patch_embed)
        # Adding an optional module must not change the data shuffle or Mixup stream.
        with torch.random.fork_rng(devices=[]):
            if variant == 'gate':
                gate = nn.Linear(2 * self.encoder.num_features, 1)
                nn.init.zeros_(gate.weight)
                nn.init.zeros_(gate.bias)
                self.encoder.patch_embed.sensor_gate = gate
            if variant == 'temporal':
                for i in (3, 7, 11):
                    self.encoder.blocks[i].temporal_adapter = TemporalAdapter(self.encoder.num_features)
            if variant == 'phase':
                self.phase_head = nn.Linear(3 * self.encoder.num_features, 40, bias=False)
                nn.init.zeros_(self.phase_head.weight)
        if mode == 'lora':
            for block in self.encoder.blocks:
                block.attn = QVLoRA(block.attn)

    def normalized_clip(self, x):
        if x.shape[-2:] != (224, 224):
            b, c, t, h, w = x.shape
            frames = x.transpose(1, 2).reshape(b * t, c, h, w)
            frames = F.interpolate(frames, (224, 224), mode='bilinear', align_corners=False)
            x = frames.reshape(b, t, c, 224, 224).transpose(1, 2)
        return ((x - self.mean) / self.std).contiguous()

    def forward(self, x):
        x = self.normalized_clip(x)
        if not hasattr(self.encoder.blocks[3], 'temporal_adapter') and not hasattr(self, 'phase_head'):
            return self.head(self.encoder(x))
        return self.classify_tokens(self.encoder.patch_embed(x))

    def forward_with_order_inputs(self, x):
        # The auxiliary order head sees content before temporal position is added.
        tokens = self.encoder.patch_embed(self.normalized_clip(x))
        return self.classify_tokens(tokens), tokens

    def forward_with_streams(self, x):
        """Reuse the two input streams for a training-only alignment loss."""
        depth, ir = self.encoder.patch_embed.streams(self.normalized_clip(x))
        return self.classify_tokens(depth + ir), depth, ir

    def classify_tokens(self, tokens):
        encoder = self.encoder
        x = tokens + encoder.pos_embed.to(tokens.dtype)
        for block in encoder.blocks:
            x = block(x)
            if hasattr(block, 'temporal_adapter'):
                x = block.temporal_adapter(x)
        logits = self.head(encoder.fc_norm(x.mean(1)))
        if hasattr(self, 'phase_head'):
            phases = x.reshape(x.size(0), 4, -1, x.size(-1)).mean(dim=2)
            phases = encoder.fc_norm(phases)
            signed_changes = phases.diff(dim=1).flatten(1)
            logits = logits + self.phase_head(signed_changes)
        return logits

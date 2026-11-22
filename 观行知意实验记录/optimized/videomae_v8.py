"""Domain adaptation for VideoMAE V2 using only paired depth and IR."""
import torch
from torch import nn
from torch.nn import functional as F
from videomae_model import VideoMAEClassifier


class ModalityBalance(nn.Module):
    """Use separate gains for each depth and IR feature channel."""
    def __init__(self, dim, *, clip_conditioned):
        super().__init__()
        self.prior = nn.Parameter(torch.zeros(dim))
        self.condition = None
        if clip_conditioned:
            self.condition = nn.Sequential(
                nn.LayerNorm(2 * dim), nn.Linear(2 * dim, 32), nn.GELU(), nn.Linear(32, dim))
            nn.init.zeros_(self.condition[-1].weight)
            nn.init.zeros_(self.condition[-1].bias)

    def gains(self, depth, ir):
        preference = self.prior.to(depth.dtype)
        if self.condition is not None:
            # Describe each sensor by its RMS activation over the synchronized clip.
            depth_rms = depth.float().square().mean(dim=1).clamp_min(1e-8).sqrt().log1p()
            ir_rms = ir.float().square().mean(dim=1).clamp_min(1e-8).sqrt().log1p()
            summary = torch.cat((depth_rms, ir_rms), dim=-1)
            preference = preference + self.condition(summary).to(depth.dtype)
        return .5 * preference.tanh().unsqueeze(-2)

    def forward(self, depth, ir):
        # At initialization both branches are exactly the original equal sum.
        return depth + ir + self.gains(depth, ir) * (depth - ir)


class LocalModalityBalance(nn.Module):
    """Predict per-tubelet corruption for paired depth and IR features."""
    def __init__(self, dim, *, spatial_centered=False):
        super().__init__()
        self.spatial_centered = spatial_centered
        self.quality = nn.Sequential(nn.LayerNorm(2 * dim), nn.Linear(2 * dim, 24),
                                     nn.GELU(), nn.Linear(24, 2))
        nn.init.zeros_(self.quality[-1].weight)
        nn.init.constant_(self.quality[-1].bias, -4.)

    def corruption_gain(self, corruption_logits):
        return self.gain_from_scores(corruption_logits.sigmoid())

    def gain_from_scores(self, scores):
        if self.spatial_centered:
            # Compare within each modality and time index before comparing modalities.
            spatial = scores.reshape(scores.size(0), 8, 14 * 14, 2)
            scores = (spatial - spatial.mean(dim=2, keepdim=True) - .05).clamp_min(0.)
            scores = scores.flatten(1, 2)
        bad_depth, bad_ir = scores.split(1, dim=-1)
        return .5 * (bad_ir - bad_depth)

    def fuse(self, depth, ir):
        corruption_logits = self.quality(torch.cat((depth, ir), dim=-1)).float()
        # Positive gain favors depth. Equal fault scores recover the original equal sum.
        gain = self.corruption_gain(corruption_logits)
        depth32, ir32 = depth.float(), ir.float()
        mixed = depth32 + ir32 + gain * (depth32 - ir32)
        return mixed.to(depth.dtype), corruption_logits

    def forward(self, depth, ir):
        return self.fuse(depth, ir)[0]


class LocalDepthRepair(LocalModalityBalance):
    """Predict a depth feature residual from IR under the learned local gate."""
    def __init__(self, dim):
        super().__init__(dim, spatial_centered=True)
        self.repair = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 64),
                                    nn.GELU(), nn.Linear(64, dim))
        nn.init.zeros_(self.repair[-1].weight)
        nn.init.zeros_(self.repair[-1].bias)

    def repair_parts(self, depth, ir):
        fused, quality = super().fuse(depth, ir)
        residual = self.repair(ir)
        strength = (-2 * self.corruption_gain(quality)).clamp(0., 1.)
        corrected = fused.float() + strength * residual.float()
        return corrected.to(fused.dtype), quality, residual

    def fuse(self, depth, ir):
        fused, quality, _ = self.repair_parts(depth, ir)
        return fused, quality


class DualPatchEmbed(nn.Module):
    def __init__(self, source, *, single_modality=None):
        super().__init__()
        self.single_modality = single_modality
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

    def depth_tokens(self, x):
        depth = self.depth(x[:, :3]).flatten(2).transpose(1, 2)
        return depth + self.depth_adapter(depth)

    def ir_tokens(self, x):
        ir = self.ir(x[:, 3:]).flatten(2).transpose(1, 2)
        return ir + self.ir_adapter(ir)

    def streams(self, x):
        return self.depth_tokens(x), self.ir_tokens(x)

    def forward(self, x):
        if self.single_modality == 'depth':
            return self.depth_tokens(x)
        if self.single_modality == 'ir':
            return self.ir_tokens(x)
        depth, ir = self.streams(x)
        if hasattr(self, 'modality_balance'):
            return self.modality_balance(depth, ir)
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
        single = {'depthonly': 'depth', 'ironly': 'ir'}.get(variant)
        self.encoder.patch_embed = DualPatchEmbed(self.encoder.patch_embed, single_modality=single)
        # Adding an optional module must not change the data shuffle or Mixup stream.
        with torch.random.fork_rng(devices=[]):
            if variant in ('balance', 'film'):
                self.encoder.patch_embed.modality_balance = ModalityBalance(
                    self.encoder.num_features, clip_conditioned=variant == 'film')
            if variant in ('local', 'localcontrast'):
                self.encoder.patch_embed.modality_balance = LocalModalityBalance(
                    self.encoder.num_features, spatial_centered=variant == 'localcontrast')
            if variant == 'localrepair':
                self.encoder.patch_embed.modality_balance = LocalDepthRepair(self.encoder.num_features)
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

    def forward_with_modality_mask(self, x, keep_depth, keep_ir):
        normalized = self.normalized_clip(x)
        depth, ir = self.encoder.patch_embed.streams(normalized)
        depth_scale = keep_depth.to(depth.dtype)[:, None, None]
        ir_scale = keep_ir.to(ir.dtype)[:, None, None]
        tokens = depth * depth_scale + ir * ir_scale
        return self.classify_tokens(tokens, position_dtype=normalized.dtype)

    def forward_with_local_quality(self, x):
        patch = self.encoder.patch_embed
        normalized = self.normalized_clip(x)
        depth, ir = patch.streams(normalized)
        fused, quality = patch.modality_balance.fuse(depth, ir)
        return self.classify_tokens(fused, position_dtype=normalized.dtype), quality

    def forward_with_cross_repair(self, x):
        patch = self.encoder.patch_embed
        normalized = self.normalized_clip(x)
        depth, ir = patch.streams(normalized)
        fused, _, residual = patch.modality_balance.repair_parts(depth, ir)
        return self.classify_tokens(fused, position_dtype=normalized.dtype), depth, residual

    def forward_with_order_inputs(self, x):
        # The auxiliary order head sees content before temporal position is added.
        tokens = self.encoder.patch_embed(self.normalized_clip(x))
        return self.classify_tokens(tokens), tokens

    def forward_with_streams(self, x):
        """Reuse the two input streams for a training-only alignment loss."""
        depth, ir = self.encoder.patch_embed.streams(self.normalized_clip(x))
        return self.classify_tokens(depth + ir), depth, ir

    def classify_tokens(self, tokens, *, position_dtype=None):
        encoder = self.encoder
        x = tokens + encoder.pos_embed.to(position_dtype or tokens.dtype)
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

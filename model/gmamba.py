import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mamba_ssm import Mamba as MambaSSM
except Exception:
    MambaSSM = None


def conv_init(conv):
    nn.init.kaiming_normal_(conv.weight, mode="fan_out")
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)


def linear_init(fc, std=0.02):
    nn.init.normal_(fc.weight, 0, std)
    if fc.bias is not None:
        nn.init.constant_(fc.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


def init_module(module):
    for m in module.modules():
        if isinstance(m, (nn.Conv1d, nn.Conv2d)):
            conv_init(m)
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            bn_init(m, 1)
        elif isinstance(m, nn.Linear):
            linear_init(m)


def edge2mat(link, num_node):
    A = np.zeros((num_node, num_node), dtype=np.float32)
    for i, j in link:
        A[j, i] = 1.0
    return A


def normalize_digraph(A):
    degree = np.sum(A, axis=0)
    Dn = np.zeros((A.shape[1], A.shape[1]), dtype=np.float32)
    for i, value in enumerate(degree):
        if value > 0:
            Dn[i, i] = value ** -1
    return np.dot(A, Dn).astype(np.float32)


def build_emotion_gait_pose_A(num_point=16):
    if num_point != 16:
        raise ValueError("Emotion-Gait data-aware adjacency expects 16 joints.")

    self_link = [(i, i) for i in range(num_point)]
    inward = [
        (0, 1),
        (1, 2),
        (2, 3),
        (2, 4),
        (4, 5),
        (5, 6),
        (2, 7),
        (7, 8),
        (8, 9),
        (0, 10),
        (10, 11),
        (11, 12),
        (0, 13),
        (13, 14),
        (14, 15),
    ]
    outward = [(j, i) for i, j in inward]
    return np.stack(
        (
            edge2mat(self_link, num_point),
            normalize_digraph(edge2mat(inward, num_point)),
            normalize_digraph(edge2mat(outward, num_point)),
        )
    )


def build_chain_pose_A(num_point):
    self_link = [(i, i) for i in range(num_point)]
    inward = [(i, i + 1) for i in range(num_point - 1)]
    outward = [(j, i) for i, j in inward]
    return np.stack(
        (
            edge2mat(self_link, num_point),
            normalize_digraph(edge2mat(inward, num_point)),
            normalize_digraph(edge2mat(outward, num_point)),
        )
    ).astype(np.float32)


def build_adjacency(num_point):
    if num_point == 16:
        return build_emotion_gait_pose_A(num_point)
    return build_chain_pose_A(num_point)


class AdaptiveGCN(nn.Module):
    def __init__(self, in_channels, out_channels, A):
        super().__init__()
        self.num_subset = A.shape[0]
        self.A = nn.Parameter(torch.from_numpy(A.astype(np.float32)), requires_grad=True)
        self.alpha = nn.Parameter(torch.zeros(self.num_subset))
        self.convs = nn.ModuleList(
            [nn.Conv2d(in_channels, out_channels, 1, bias=False) for _ in range(self.num_subset)]
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(True)
        init_module(self)

    def forward(self, x):
        y = None
        for subset_idx in range(self.num_subset):
            adj = self.A[subset_idx]
            identity = torch.eye(adj.size(-1), device=adj.device, dtype=adj.dtype)
            mixed_adj = adj + self.alpha[subset_idx] * identity
            z = torch.einsum("nctu,vu->nctv", x, mixed_adj)
            z = self.convs[subset_idx](z)
            y = z if y is None else y + z
        return self.relu(self.bn(y))


class MambaLite(nn.Module):
    """Fallback when mamba_ssm is unavailable."""

    def __init__(self, d_model, d_conv=4, expand=2):
        super().__init__()
        inner = int(d_model * expand)
        self.in_proj = nn.Linear(d_model, inner * 2)
        self.dwconv = nn.Conv1d(
            inner,
            inner,
            kernel_size=d_conv,
            padding=max(0, d_conv - 1),
            groups=inner,
            bias=True,
        )
        self.dt_proj = nn.Linear(inner, inner)
        self.out_proj = nn.Linear(inner, d_model)
        init_module(self)

    def forward(self, x):
        seq_len = x.size(1)
        hidden, gate = self.in_proj(x).chunk(2, dim=-1)
        hidden = self.dwconv(hidden.transpose(1, 2))[..., :seq_len].transpose(1, 2)
        hidden = F.silu(hidden)
        delta = torch.sigmoid(self.dt_proj(hidden))

        state = torch.zeros_like(hidden[:, 0])
        outputs = []
        for step in range(seq_len):
            state = delta[:, step] * state + (1.0 - delta[:, step]) * hidden[:, step]
            outputs.append(state * torch.sigmoid(gate[:, step]))
        y = torch.stack(outputs, dim=1)
        return self.out_proj(y)


class CrossStreamsGate(nn.Module):
    def __init__(self, channels, hidden_ratio=0.5, drop_out=0.0):
        super().__init__()
        hidden = max(int(channels * hidden_ratio), 32)
        self.score = nn.Sequential(
            nn.Linear(channels * 4, hidden),
            nn.ReLU(True),
            nn.Dropout(drop_out) if drop_out else nn.Identity(),
            nn.Linear(hidden, 2),
        )
        self.channel_gate = nn.Sequential(
            nn.Linear(channels * 4, hidden),
            nn.ReLU(True),
            nn.Linear(hidden, channels * 2),
            nn.Sigmoid(),
        )
        init_module(self)
        nn.init.constant_(self.score[-1].weight, 0)
        nn.init.constant_(self.score[-1].bias, 0)

    def forward(self, pose_feat, motion_feat):
        pose_pool = pose_feat.mean(dim=(-1, -2))
        motion_pool = motion_feat.mean(dim=(-1, -2))
        context = torch.cat(
            [
                pose_pool,
                motion_pool,
                torch.abs(pose_pool - motion_pool),
                pose_pool * motion_pool,
            ],
            dim=1,
        )
        stream_weights = torch.softmax(self.score(context), dim=1)
        gate_pose, gate_motion = self.channel_gate(context).chunk(2, dim=1)
        gate_pose = gate_pose.unsqueeze(-1).unsqueeze(-1)
        gate_motion = gate_motion.unsqueeze(-1).unsqueeze(-1)

        pose_out = pose_feat + stream_weights[:, 1:2, None, None] * gate_pose * motion_feat
        motion_out = motion_feat + stream_weights[:, 0:1, None, None] * gate_motion * pose_feat
        return pose_out, motion_out, stream_weights


class MTM(nn.Module):
    """Multi-scale Temporal Modeling."""

    def __init__(self, channels, kernel_sizes=(5,), drop_out=0.0):
        super().__init__()
        kernel_sizes = [int(k) for k in kernel_sizes]
        if not kernel_sizes:
            raise ValueError("kernel_sizes must not be empty.")
        for kernel_size in kernel_sizes:
            if kernel_size < 1 or kernel_size % 2 == 0:
                raise ValueError("MTM kernel sizes must be positive odd integers.")

        self.channels = int(channels)
        self.kernel_sizes = kernel_sizes
        self.drop = nn.Dropout(drop_out) if drop_out else nn.Identity()
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(self.channels, self.channels, 1, bias=False),
                    nn.BatchNorm2d(self.channels),
                )
                for _ in self.kernel_sizes
            ]
        )
        self.act = nn.GELU()

        init_module(self.branches)
        for branch in self.branches:
            bn_init(branch[-1], 0)

    @staticmethod
    def _shift_temporal(x, offset):
        if offset == 0:
            return x
        if abs(offset) >= x.size(2):
            return torch.zeros_like(x)
        if offset > 0:
            shifted = x[:, :, offset:, :]
            return F.pad(shifted, (0, 0, 0, offset))
        offset = abs(offset)
        shifted = x[:, :, :-offset, :]
        return F.pad(shifted, (0, 0, offset, 0))

    def _cycle_shift(self, x, kernel_size):
        if kernel_size == 1:
            return x

        radius = kernel_size // 2
        y = x.new_zeros(x.size())
        for group_idx, offset in enumerate(range(-radius, radius + 1)):
            channels = list(range(group_idx, self.channels, kernel_size))
            if channels:
                shifted = self._shift_temporal(x, offset)
                y[:, channels, :, :] = shifted[:, channels, :, :]
        return y

    def forward(self, x):
        y = x
        for kernel_size, branch in zip(self.kernel_sizes, self.branches):
            y = y + self.drop(branch(self._cycle_shift(x, kernel_size)))
        return self.act(y)


class BiMambaBlock(nn.Module):
    """MTM + Bidirection-Mamba."""

    def __init__(
        self,
        channels,
        d_state=16,
        d_conv=4,
        expand=2,
        dropout=0.0,
        use_mamba_ssm=True,
        channel_ratio=0.5,
        mtm_kernel_sizes=(5,),
    ):
        super().__init__()
        channels = int(channels)
        branch_channels = max(1, int(round(channels * float(channel_ratio))))
        self.channels = channels
        self.branch_channels = branch_channels
        self.uses_mamba_ssm = bool(use_mamba_ssm and MambaSSM is not None)

        self.in_norm = nn.LayerNorm(channels)
        self.channel_proj = nn.Sequential(
            nn.Conv2d(channels, branch_channels, 1, bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(True),
        )
        self.mtm = MTM(branch_channels, kernel_sizes=mtm_kernel_sizes, drop_out=dropout)

        self.fwd_norm = nn.LayerNorm(branch_channels)
        self.bwd_norm = nn.LayerNorm(branch_channels)
        self.mamba_fwd = self._make_mamba(branch_channels, d_state, d_conv, expand)
        self.mamba_bwd = self._make_mamba(branch_channels, d_state, d_conv, expand)
        self.fuse_norm = nn.LayerNorm(branch_channels * 2)
        self.out_proj = nn.Linear(branch_channels * 2, channels, bias=False)
        self.out_bn = nn.BatchNorm2d(channels)
        self.drop = nn.Dropout(dropout) if dropout else nn.Identity()

        init_module(self.channel_proj)
        init_module(self.out_proj)
        bn_init(self.out_bn, 1)

    def _make_mamba(self, channels, d_state, d_conv, expand):
        if self.uses_mamba_ssm:
            return MambaSSM(
                d_model=channels,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
        return MambaLite(channels, d_conv=d_conv, expand=expand)

    @staticmethod
    def _apply_channel_norm(x, norm):
        return norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()

    def _run_branch(self, seq, norm, mamba):
        return seq + self.drop(mamba(norm(seq)))

    def forward(self, x):
        n, c, t, v = x.size()
        h = self._apply_channel_norm(x, self.in_norm)
        h = self.channel_proj(h)
        h = self.mtm(h)

        seq = h.permute(0, 3, 2, 1).contiguous().view(n * v, t, self.branch_channels)
        seq_f = self._run_branch(seq, self.fwd_norm, self.mamba_fwd)

        seq_b = torch.flip(seq, dims=(1,))
        seq_b = self._run_branch(seq_b, self.bwd_norm, self.mamba_bwd)
        seq_b = torch.flip(seq_b, dims=(1,))

        seq = torch.cat([seq_f, seq_b], dim=-1)
        seq = self.out_proj(self.fuse_norm(seq))
        y = seq.view(n, v, t, c).permute(0, 3, 2, 1).contiguous()
        y = self.out_bn(y)
        return x + self.drop(y)


class DSAW(nn.Module):
    def __init__(self, channels, hidden_ratio=0.25, drop_prob=0.0, preserve_magnitude=True):
        super().__init__()
        hidden = max(int(channels * hidden_ratio), 16)
        self.preserve_magnitude = preserve_magnitude
        self.score = nn.Sequential(
            nn.Linear(channels * 4, hidden, bias=False),
            nn.ReLU(True),
            nn.Dropout(drop_prob),
            nn.Linear(hidden, 2, bias=True),
        )
        init_module(self.score)
        nn.init.constant_(self.score[-1].weight, 0)
        nn.init.constant_(self.score[-1].bias, 0)

    def forward(self, pose_feat, motion_feat):
        joint_context = torch.cat(
            [
                pose_feat,
                motion_feat,
                torch.abs(pose_feat - motion_feat),
                pose_feat * motion_feat,
            ],
            dim=1,
        )
        weights = torch.softmax(self.score(joint_context), dim=1)
        scale = weights.size(1) if self.preserve_magnitude else 1.0
        pose_weighted = pose_feat * weights[:, 0:1] * scale
        motion_weighted = motion_feat * weights[:, 1:2] * scale
        return pose_weighted, motion_weighted, weights


class DynamicMultiTaskHead(nn.Module):
    def __init__(
        self,
        final_channels,
        num_class=4,
        aff_dim=1488,
        drop_out=0.0,
        adaptive_weight_hidden_ratio=0.25,
        adaptive_weight_drop=0.0,
        head_hidden_ratio=0.75,
        aff_loss_type="MSE",
        aff_loss_ramp_epochs=10,
        class_loss_weights=None,
        class_loss_smoothing=0.0,
    ):
        super().__init__()
        self.aff_loss_type = str(aff_loss_type)
        self.aff_loss_ramp_epochs = int(aff_loss_ramp_epochs)
        self.class_loss_smoothing = float(class_loss_smoothing)
        if self.class_loss_smoothing < 0 or self.class_loss_smoothing >= 1:
            raise ValueError("class_loss_smoothing must be in [0, 1).")
        if class_loss_weights is None:
            class_loss_weights = []
        class_loss_weights = torch.as_tensor(class_loss_weights, dtype=torch.float32)
        if class_loss_weights.numel() not in (0, int(num_class)):
            raise ValueError(
                "class_loss_weights must be empty or have num_class={} values.".format(num_class)
            )
        self.register_buffer("class_loss_weights", class_loss_weights)

        self.stream_weight = DSAW(
            final_channels,
            hidden_ratio=adaptive_weight_hidden_ratio,
            drop_prob=adaptive_weight_drop,
        )

        context_dim = final_channels * 3
        head_hidden = max(int(context_dim * head_hidden_ratio), final_channels)

        self.out_drop = nn.Dropout(drop_out) if drop_out else nn.Identity()
        self.context_norm = nn.LayerNorm(context_dim)
        self.classifier = nn.Linear(context_dim, num_class)
        self.pose_classifier = nn.Linear(final_channels, num_class)
        self.motion_classifier = nn.Linear(final_channels, num_class)
        self.aff_head = nn.Sequential(
            nn.Linear(context_dim, head_hidden),
            nn.ReLU(True),
            nn.Dropout(drop_out),
            nn.Linear(head_hidden, aff_dim),
        )

        linear_init(self.classifier, math.sqrt(2.0 / context_dim))
        linear_init(self.pose_classifier, math.sqrt(2.0 / final_channels))
        linear_init(self.motion_classifier, math.sqrt(2.0 / final_channels))
        init_module(self.aff_head)

    def forward(self, pose_repr, motion_repr, fused_repr):
        pose_weighted, motion_weighted, stream_weights = self.stream_weight(
            pose_repr,
            motion_repr,
        )
        final_context = torch.cat([pose_weighted, motion_weighted, fused_repr], dim=1)
        final_context = self.context_norm(final_context)
        final_context = self.out_drop(final_context)

        logits = self.classifier(final_context)
        pose_logits = self.pose_classifier(pose_repr)
        motion_logits = self.motion_classifier(motion_repr)
        aff_pred = self.aff_head(final_context)

        aux = {
            "stream_weights": stream_weights,
            "final_context": final_context,
            "pose_weighted": pose_weighted,
            "motion_weighted": motion_weighted,
            "pose_logits": pose_logits,
            "motion_logits": motion_logits,
        }
        return logits, aff_pred, aux

    @staticmethod
    def parse_model_output(output):
        if not isinstance(output, (tuple, list)):
            return output, None, {}
        logits = output[0]
        aff_pred = output[1] if len(output) > 1 else None
        aux = output[2] if len(output) > 2 and isinstance(output[2], dict) else {}
        return logits, aff_pred, aux

    def get_aff_loss_scale(self, epoch):
        ramp_epochs = int(self.aff_loss_ramp_epochs)
        if ramp_epochs <= 0:
            return 1.0
        return min(1.0, float(epoch + 1) / ramp_epochs)

    def compute_loss(
        self,
        output,
        label,
        feature,
        epoch,
        training=False,
        shuffle_affective_target=False,
    ):
        logits, aff_pred, aux = self.parse_model_output(output)
        feature = feature.view(feature.size(0), -1)
        if shuffle_affective_target and training and feature.size(0) > 1:
            feature = feature[torch.randperm(feature.size(0), device=feature.device)]

        class_weights = self.class_loss_weights
        if class_weights.numel() == 0:
            class_weights = None
        else:
            if class_weights.numel() != logits.size(1):
                raise ValueError(
                    "class_loss_weights has {} values, but logits has {} classes.".format(
                        class_weights.numel(),
                        logits.size(1),
                    )
                )
            class_weights = class_weights.to(device=logits.device, dtype=logits.dtype)
        cls_vec = F.cross_entropy(
            logits,
            label,
            weight=class_weights,
            label_smoothing=self.class_loss_smoothing,
            reduction="none",
        )
        if aff_pred is None:
            aff_vec = torch.zeros_like(cls_vec)
        else:
            aff_pred = aff_pred.view(aff_pred.size(0), -1)
            if aff_pred.shape != feature.shape:
                raise ValueError(
                    "Affective output shape {} does not match target shape {}.".format(
                        tuple(aff_pred.shape),
                        tuple(feature.shape),
                    )
                )
            if self.aff_loss_type == "SmoothL1":
                aff_vec = F.smooth_l1_loss(aff_pred, feature, reduction="none").mean(dim=1)
            else:
                aff_vec = F.mse_loss(aff_pred, feature, reduction="none").mean(dim=1)
        cls_weight = 0.7
        aff_weight = 0.3
        weights = logits.new_tensor([cls_weight, aff_weight]).view(1, 2).expand(logits.size(0), 2)

        aff_scale = self.get_aff_loss_scale(epoch)
        loss_vec = weights[:, 0] * cls_vec + weights[:, 1] * (aff_scale * aff_vec)
        loss = loss_vec.mean()

        metrics = {
            "loss_cls": cls_vec.mean(),
            "loss_aff": aff_vec.mean(),
            "loss_weight_cls": weights[:, 0].mean(),
            "loss_weight_aff": weights[:, 1].mean(),
            "aff_scale": logits.new_tensor(aff_scale),
        }
        contrastive_loss = aux.get("contrastive_loss", None)
        if contrastive_loss is not None:
            contrastive_loss = contrastive_loss.to(device=logits.device, dtype=logits.dtype)
            loss = loss + contrastive_loss
            metrics["loss_contrastive"] = contrastive_loss.detach()
            if aux.get("contrastive_loss_pose", None) is not None:
                metrics["loss_contrastive_pose"] = aux["contrastive_loss_pose"].detach()
            if aux.get("contrastive_loss_motion", None) is not None:
                metrics["loss_contrastive_motion"] = aux["contrastive_loss_motion"].detach()
        return logits, loss, metrics


class EmoConditioning(nn.Module):
    def __init__(
        self,
        feature_dim,
        channels,
        hidden_ratio=0.25,
        drop_out=0.0,
        init_scale=0.05,
    ):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.channels = int(channels)
        hidden = max(int(self.feature_dim * hidden_ratio), self.channels)
        self.norm = nn.LayerNorm(self.feature_dim)
        self.encoder = nn.Sequential(
            nn.Linear(self.feature_dim, hidden),
            nn.ReLU(True),
            nn.Dropout(drop_out) if drop_out else nn.Identity(),
            nn.Linear(hidden, self.channels * 6),
        )
        self.gamma = nn.Parameter(torch.tensor(float(init_scale)))
        init_module(self.encoder)
        nn.init.constant_(self.encoder[-1].weight, 0)
        nn.init.constant_(self.encoder[-1].bias, 0)

    def forward(self, pose_repr, motion_repr, fused_repr, emotion_feat=None):
        if emotion_feat is None:
            return pose_repr, motion_repr, fused_repr, None

        emotion_feat = emotion_feat.view(emotion_feat.size(0), -1).to(
            device=pose_repr.device,
            dtype=pose_repr.dtype,
        )
        if emotion_feat.size(1) != self.feature_dim:
            raise ValueError(
                "emotion feature dim {} does not match configured dim {}.".format(
                    emotion_feat.size(1),
                    self.feature_dim,
                )
            )

        params = self.encoder(self.norm(emotion_feat)).view(emotion_feat.size(0), 6, self.channels)
        gates = torch.tanh(params[:, :3])
        biases = params[:, 3:]
        scale = self.gamma
        pose_repr = pose_repr * (1.0 + scale * gates[:, 0]) + scale * biases[:, 0]
        motion_repr = motion_repr * (1.0 + scale * gates[:, 1]) + scale * biases[:, 1]
        fused_repr = fused_repr * (1.0 + scale * gates[:, 2]) + scale * biases[:, 2]

        aux = {
            "emotion_gates": gates,
            "emotion_biases": biases,
            "emotion_feature": emotion_feat,
        }
        return pose_repr, motion_repr, fused_repr, aux


class PartAwareGate(nn.Module):
    def __init__(
        self,
        channels,
        num_joints=16,
        hidden_ratio=0.25,
        drop_out=0.0,
        init_scale=0.1,
    ):
        super().__init__()
        self.channels = int(channels)
        self.part_groups = self.get_part_groups(int(num_joints))
        hidden = max(int(self.channels * hidden_ratio), 16)
        self.part_mlp = nn.Sequential(
            nn.Conv1d(self.channels, hidden, 1, bias=False),
            nn.BatchNorm1d(hidden),
            nn.ReLU(True),
            nn.Dropout(drop_out) if drop_out else nn.Identity(),
            nn.Conv1d(hidden, self.channels, 1, bias=True),
        )
        self.gamma = nn.Parameter(torch.tensor(float(init_scale)))
        init_module(self.part_mlp)
        nn.init.constant_(self.part_mlp[-1].weight, 0)
        nn.init.constant_(self.part_mlp[-1].bias, 0)

    @staticmethod
    def get_part_groups(num_joints):
        if num_joints == 16:
            return [
                [0, 1, 2, 3],
                [4, 5, 6],
                [7, 8, 9],
                [10, 11, 12],
                [13, 14, 15],
            ]

        chunk = max(1, int(math.ceil(float(num_joints) / 5.0)))
        groups = []
        for start in range(0, num_joints, chunk):
            groups.append(list(range(start, min(start + chunk, num_joints))))
        return groups

    def forward(self, x):
        n, c, _, v = x.size()
        descriptors = []
        for group in self.part_groups:
            valid_group = [idx for idx in group if idx < v]
            if valid_group:
                descriptors.append(x[:, :, :, valid_group].mean(dim=(2, 3)))
            else:
                descriptors.append(x.new_zeros(n, c))
        descriptors = torch.stack(descriptors, dim=-1)
        part_logits = self.part_mlp(descriptors)

        gate = x.new_zeros(n, c, v)
        for part_idx, group in enumerate(self.part_groups):
            valid_group = [idx for idx in group if idx < v]
            if valid_group:
                part_gate = part_logits[:, :, part_idx].unsqueeze(-1)
                gate[:, :, valid_group] = part_gate.expand(-1, -1, len(valid_group))
        gate = 1.0 + self.gamma * torch.tanh(gate)
        return x * gate.unsqueeze(2)


class SpatialTemporalAttentionPool(nn.Module):
    def __init__(self, channels, hidden_ratio=0.25, drop_out=0.0):
        super().__init__()
        channels = int(channels)
        hidden = max(int(channels * hidden_ratio), 16)
        self.score = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(True),
            nn.Dropout(drop_out) if drop_out else nn.Identity(),
            nn.Conv2d(hidden, 1, 1, bias=True),
        )
        init_module(self.score)
        nn.init.constant_(self.score[-1].weight, 0)
        nn.init.constant_(self.score[-1].bias, 0)

    def forward(self, x, n, m):
        _, c, t, v = x.size()
        features = x.view(n, m, c, t, v)
        logits = self.score(x).view(n, m, 1, t, v)
        weights = torch.softmax(logits.view(n, 1, -1), dim=-1).view(n, m, 1, t, v)
        return (features * weights).sum(dim=(1, 3, 4))


class RenovateNet(nn.Module):
    def __init__(
        self,
        n_channel,
        n_class,
        alp=0.125,
        tmp=0.125,
        mom=0.9,
        h_channel=None,
        pred_threshold=0.0,
        use_p_map=True,
    ):
        super().__init__()
        self.n_class = int(n_class)
        self.alp = float(alp)
        self.tmp = float(tmp)
        self.mom = float(mom)
        self.pred_threshold = float(pred_threshold)
        self.use_p_map = bool(use_p_map)
        self.h_channel = int(h_channel or n_channel)
        self.cl_fc = nn.Linear(n_channel, self.h_channel)
        self.loss = nn.CrossEntropyLoss()
        self.register_buffer("avg_f", torch.randn(self.h_channel, self.n_class))
        init_module(self.cl_fc)

    def onehot(self, label):
        return F.one_hot(label, self.n_class).float()

    def get_mask_fn_fp(self, lbl_one, pred_one, logit):
        tp = lbl_one * pred_one
        fn = lbl_one - tp
        fp = pred_one - tp
        tp = tp * (logit > self.pred_threshold).float()
        has_fn = (fn.sum(0) > 1e-8).float().unsqueeze(1)
        has_fp = (fp.sum(0) > 1e-8).float().unsqueeze(1)
        return tp, fn, fp, has_fn, has_fp

    def local_avg_tp_fn_fp(self, feature, mask, fn, fp):
        feature = feature.permute(1, 0)
        avg_f = self.avg_f.detach()
        f_fn = torch.matmul(feature, F.normalize(fn, p=1, dim=0))
        f_fp = torch.matmul(feature, F.normalize(fp, p=1, dim=0))
        mask_sum = mask.sum(0, keepdim=True)
        f_mask = torch.matmul(feature, mask) / (mask_sum + 1e-12)
        has_obj = (mask_sum > 1e-8).float()
        keep = torch.where(has_obj > 0.1, torch.full_like(has_obj, self.mom), torch.ones_like(has_obj))
        f_mem = avg_f * keep + (1.0 - keep) * f_mask
        with torch.no_grad():
            self.avg_f.copy_(f_mem.detach())
        return f_mem, f_fn, f_fp

    def get_score(self, feature, lbl_one, logit, f_mem, f_fn, f_fp, s_fn, s_fp):
        feature = F.normalize(feature, dim=1)
        f_mem = F.normalize(f_mem.permute(1, 0), dim=1)
        f_fn = F.normalize(f_fn.permute(1, 0), dim=1)
        f_fp = F.normalize(f_fp.permute(1, 0), dim=1)
        p_map = ((1.0 - logit) * lbl_one * self.alp) if self.use_p_map else (lbl_one * self.alp)
        score_mem = torch.matmul(f_mem, feature.T)
        score_fn = torch.matmul(f_fn, feature.T) - 1.0
        score_fp = -torch.matmul(f_fp, feature.T) - 1.0
        fn_map = score_fn * p_map.T * s_fn
        fp_map = score_fp * p_map.T * s_fp
        return (score_mem + fn_map) / self.tmp, (score_mem + fp_map) / self.tmp

    def forward(self, feature, label, logit, return_loss=True):
        feature = self.cl_fc(feature)
        pred = logit.argmax(1)
        lbl_one = self.onehot(label)
        pred_one = self.onehot(pred)
        logit_soft = logit.softmax(1)
        mask, fn, fp, has_fn, has_fp = self.get_mask_fn_fp(lbl_one, pred_one, logit_soft)
        f_mem, f_fn, f_fp = self.local_avg_tp_fn_fp(feature, mask, fn, fp)
        score_fn, score_fp = self.get_score(feature, lbl_one, logit_soft, f_mem, f_fn, f_fp, has_fn, has_fp)
        if return_loss:
            return 0.5 * (self.loss(score_fn.T, label) + self.loss(score_fp.T, label))
        return score_fn.T.contiguous(), score_fp.T.contiguous()


class PRC(nn.Module):
    def __init__(self, n_channel, n_frame, n_joint, n_person, n_class=4, h_channel=256, **kwargs):
        super().__init__()
        self.n_channel = int(n_channel)
        self.n_frame = int(n_frame)
        self.n_joint = int(n_joint)
        self.n_person = int(n_person)
        spatial_channels = max(1, h_channel // self.n_joint)
        temporal_channels = max(1, h_channel // self.n_frame)
        self.spatio_cl_net = RenovateNet(
            n_channel=spatial_channels * self.n_joint,
            h_channel=h_channel,
            n_class=n_class,
            **kwargs,
        )
        self.tempor_cl_net = RenovateNet(
            n_channel=temporal_channels * self.n_frame,
            h_channel=h_channel,
            n_class=n_class,
            **kwargs,
        )
        self.spatio_squeeze = nn.Sequential(
            nn.Conv2d(self.n_channel, spatial_channels, 1),
            nn.BatchNorm2d(spatial_channels),
            nn.ReLU(True),
        )
        self.tempor_squeeze = nn.Sequential(
            nn.Conv2d(self.n_channel, temporal_channels, 1),
            nn.BatchNorm2d(temporal_channels),
            nn.ReLU(True),
        )
        init_module(self.spatio_squeeze)
        init_module(self.tempor_squeeze)

    def forward(self, raw_feat, label, logit, **kwargs):
        raw_feat = raw_feat.view(-1, self.n_person, self.n_channel, self.n_frame, self.n_joint)
        spatio_feat = raw_feat.mean(1).mean(-2, keepdim=True)
        spatio_feat = self.spatio_squeeze(spatio_feat).flatten(1)
        spatio_cl_loss = self.spatio_cl_net(spatio_feat, label, logit, **kwargs)

        tempor_feat = raw_feat.mean(1).mean(-1, keepdim=True)
        tempor_feat = self.tempor_squeeze(tempor_feat).flatten(1)
        tempor_cl_loss = self.tempor_cl_net(tempor_feat, label, logit, **kwargs)
        return spatio_cl_loss + tempor_cl_loss


class HGMBlock(nn.Module):
    """Hybrid Graph-Mamba block: AdaptiveGCN + PartAwareGate + MTM + Bi-Mamba + FFN."""

    def __init__(
        self,
        in_channels,
        out_channels,
        A,
        num_joints=16,
        stride=1,
        residual=True,
        mamba_d_state=16,
        mamba_d_conv=4,
        mamba_expand=2,
        drop_out=0.0,
        use_mamba_ssm=True,
        ffn_ratio=2.0,
        mamba_channel_ratio=0.5,
        mamba_mtm_kernel_sizes=(5,),
        use_part_gate=True,
        part_gate_hidden_ratio=0.25,
        part_gate_init_scale=0.1,
    ):
        super().__init__()
        self.gcn = AdaptiveGCN(in_channels, out_channels, A)
        self.temporal_conv = nn.Sequential(
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=(3, 1),
                stride=(stride, 1),
                padding=(1, 0),
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )
        self.part_gate = (
            PartAwareGate(
                out_channels,
                num_joints=num_joints,
                hidden_ratio=part_gate_hidden_ratio,
                drop_out=drop_out,
                init_scale=part_gate_init_scale,
            )
            if use_part_gate
            else nn.Identity()
        )
        self.mamba = BiMambaBlock(
            out_channels,
            d_state=mamba_d_state,
            d_conv=mamba_d_conv,
            expand=mamba_expand,
            dropout=drop_out,
            use_mamba_ssm=use_mamba_ssm,
            channel_ratio=mamba_channel_ratio,
            mtm_kernel_sizes=mamba_mtm_kernel_sizes,
        )

        hidden = max(int(out_channels * ffn_ratio), out_channels)
        self.ffn = nn.Sequential(
            nn.Conv2d(out_channels, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
            nn.Dropout(drop_out) if drop_out else nn.Identity(),
            nn.Conv2d(hidden, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.relu = nn.ReLU(True)

        if not residual:
            self.residual = lambda x: 0
        elif in_channels == out_channels and stride == 1:
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=(stride, 1), bias=False),
                nn.BatchNorm2d(out_channels),
            )
            init_module(self.residual)

        init_module(self.temporal_conv)
        init_module(self.ffn)
        if isinstance(self.ffn[-1], nn.BatchNorm2d):
            bn_init(self.ffn[-1], 0)

    def forward(self, x):
        y = self.gcn(x)
        y = self.temporal_conv(y)
        y = self.part_gate(y)
        y = self.mamba(y)
        y = y + self.ffn(y)
        return self.relu(y + self.residual(x))


class GMamba(nn.Module):
    def __init__(
        self,
        num_class=4,
        num_point=16,
        in_channels_p=3,
        in_channels_m=8,
        stage_dims=(64, 64, 128, 256),
        stage_strides=(1, 1, 2, 1),
        fusion_after=(2, 4),
        aff_dim=1488,
        drop_out=0.25,
        mamba_d_state=16,
        mamba_d_conv=4,
        mamba_expand=2,
        use_mamba_ssm=True,
        fusion_hidden_ratio=0.5,
        ffn_ratio=2.0,
        adaptive_weight_hidden_ratio=0.25,
        adaptive_weight_drop=0.0,
        head_hidden_ratio=0.75,
        aff_loss_type="MSE",
        aff_loss_ramp_epochs=10,
        class_loss_weights=None,
        class_loss_smoothing=0.0,
        use_emotion_feature=True,
        emotion_feature_dim=None,
        emotion_hidden_ratio=0.25,
        emotion_init_scale=0.05,
        mamba_channel_ratio=0.5,
        mamba_mtm_kernel_sizes=(5,),
        use_part_gate=True,
        part_gate_hidden_ratio=0.25,
        part_gate_init_scale=0.1,
        use_st_attention_pool=True,
        pool_hidden_ratio=0.25,
        use_layer_contrastive=True,
        contrastive_weight=0.05,
        contrastive_stage_weights=None,
        contrastive_h_channel=256,
        contrastive_alp=0.125,
        contrastive_tmp=0.125,
        contrastive_mom=0.9,
        contrastive_pred_threshold=0.0,
        contrastive_use_p_map=True,
        contrastive_n_person=1,
        **kwargs
    ):
        super().__init__()
        self.num_class = num_class
        self.num_point = num_point
        self.stage_dims = list(stage_dims)
        self.stage_strides = list(stage_strides)
        self.fusion_after = [int(point) for point in fusion_after]
        self.mamba_mtm_kernel_sizes = [int(k) for k in mamba_mtm_kernel_sizes]
        self.use_emotion_feature = bool(use_emotion_feature)
        self.emotion_feature_dim = int(emotion_feature_dim or aff_dim)
        self.use_part_gate = bool(use_part_gate)
        self.use_st_attention_pool = bool(use_st_attention_pool)
        self.use_layer_contrastive = bool(use_layer_contrastive)
        self.contrastive_weight = float(contrastive_weight)
        self.contrastive_n_person = int(contrastive_n_person)
        if contrastive_stage_weights is None:
            contrastive_stage_weights = [1.0 for _ in self.stage_dims]
        if len(contrastive_stage_weights) != len(self.stage_dims):
            raise ValueError("contrastive_stage_weights must match stage_dims length.")
        self.contrastive_stage_weights = [float(weight) for weight in contrastive_stage_weights]

        backend = "mamba_ssm" if bool(use_mamba_ssm and MambaSSM is not None) else "mamba_lite"
        self.mamba_backend = "gmamba_{}".format(backend)

        if len(self.stage_dims) != len(self.stage_strides):
            raise ValueError("stage_strides must have the same length as stage_dims.")
        if not self.stage_dims:
            raise ValueError("stage_dims must not be empty.")

        A = build_adjacency(num_point=num_point)

        self.data_bn_p = nn.BatchNorm1d(in_channels_p * num_point)
        self.data_bn_m = nn.BatchNorm1d(in_channels_m * num_point)

        self.pose_stages = nn.ModuleList()
        self.motion_stages = nn.ModuleList()
        self.stage_frames = []
        prev_pose_channels = in_channels_p
        prev_motion_channels = in_channels_m
        current_frames = int(kwargs.get("emotion_time_steps", kwargs.get("window_size", 48)))
        for stage_idx, (out_channels, stride) in enumerate(zip(self.stage_dims, self.stage_strides)):
            block_kwargs = dict(
                A=A,
                num_joints=num_point,
                stride=stride,
                residual=stage_idx != 0,
                mamba_d_state=mamba_d_state,
                mamba_d_conv=mamba_d_conv,
                mamba_expand=mamba_expand,
                drop_out=drop_out,
                use_mamba_ssm=use_mamba_ssm,
                ffn_ratio=ffn_ratio,
                mamba_channel_ratio=mamba_channel_ratio,
                mamba_mtm_kernel_sizes=self.mamba_mtm_kernel_sizes,
                use_part_gate=self.use_part_gate,
                part_gate_hidden_ratio=part_gate_hidden_ratio,
                part_gate_init_scale=part_gate_init_scale,
            )
            self.pose_stages.append(HGMBlock(prev_pose_channels, out_channels, **block_kwargs))
            self.motion_stages.append(HGMBlock(prev_motion_channels, out_channels, **block_kwargs))
            prev_pose_channels = out_channels
            prev_motion_channels = out_channels
            current_frames = int(math.ceil(float(current_frames) / float(stride)))
            self.stage_frames.append(current_frames)

        contrastive_kwargs = dict(
            n_class=num_class,
            h_channel=int(contrastive_h_channel),
            alp=contrastive_alp,
            tmp=contrastive_tmp,
            mom=contrastive_mom,
            pred_threshold=contrastive_pred_threshold,
            use_p_map=contrastive_use_p_map,
        )
        self.pose_contrastive_heads = nn.ModuleList()
        self.motion_contrastive_heads = nn.ModuleList()
        if self.use_layer_contrastive:
            for channels, frames in zip(self.stage_dims, self.stage_frames):
                self.pose_contrastive_heads.append(
                    PRC(
                        n_channel=channels,
                        n_frame=frames,
                        n_joint=num_point,
                        n_person=self.contrastive_n_person,
                        **contrastive_kwargs,
                    )
                )
                self.motion_contrastive_heads.append(
                    PRC(
                        n_channel=channels,
                        n_frame=frames,
                        n_joint=num_point,
                        n_person=self.contrastive_n_person,
                        **contrastive_kwargs,
                    )
                )

        self.fusions = nn.ModuleDict()
        for point in self.fusion_after:
            if point < 1 or point > len(self.stage_dims):
                raise ValueError("fusion_after points must be valid stage indices.")
            channels = self.stage_dims[point - 1]
            self.fusions[str(point)] = CrossStreamsGate(
                channels,
                hidden_ratio=fusion_hidden_ratio,
                drop_out=drop_out,
            )

        final_channels = self.stage_dims[-1]
        self.pose_pool = (
            SpatialTemporalAttentionPool(final_channels, hidden_ratio=pool_hidden_ratio, drop_out=drop_out)
            if self.use_st_attention_pool
            else None
        )
        self.motion_pool = (
            SpatialTemporalAttentionPool(final_channels, hidden_ratio=pool_hidden_ratio, drop_out=drop_out)
            if self.use_st_attention_pool
            else None
        )
        self.fusion_pool = (
            SpatialTemporalAttentionPool(final_channels, hidden_ratio=pool_hidden_ratio, drop_out=drop_out)
            if self.use_st_attention_pool
            else None
        )
        self.emotion_condition = (
            EmoConditioning(
                feature_dim=self.emotion_feature_dim,
                channels=final_channels,
                hidden_ratio=emotion_hidden_ratio,
                drop_out=drop_out,
                init_scale=emotion_init_scale,
            )
            if self.use_emotion_feature
            else None
        )
        self.dla_head = DynamicMultiTaskHead(
            final_channels=final_channels,
            num_class=num_class,
            aff_dim=aff_dim,
            drop_out=drop_out,
            adaptive_weight_hidden_ratio=adaptive_weight_hidden_ratio,
            adaptive_weight_drop=adaptive_weight_drop,
            head_hidden_ratio=head_hidden_ratio,
            aff_loss_type=aff_loss_type,
            aff_loss_ramp_epochs=aff_loss_ramp_epochs,
            class_loss_weights=class_loss_weights,
            class_loss_smoothing=class_loss_smoothing,
        )

        bn_init(self.data_bn_p, 1)
        bn_init(self.data_bn_m, 1)

    def _preprocess(self, x, in_channels, bn):
        n, _, t, v, m = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(n, m * v * in_channels, t)
        x = bn(x)
        x = x.view(n, m, v, in_channels, t).permute(0, 1, 3, 4, 2).contiguous()
        return x.view(n * m, in_channels, t, v)

    @staticmethod
    def _global_pool(x, n, m, pool=None):
        if pool is not None:
            return pool(x, n, m)
        return x.view(n, m, x.size(1), -1).mean(3).mean(1)

    def forward(
        self,
        x_p,
        x_m,
        label=None,
        emotion_feat=None,
        get_hidden_feat=False,
        collect_visuals=False,
        **kwargs
    ):
        if emotion_feat is None:
            emotion_feat = kwargs.get("feature", None)
        n, c_p, _, _, m = x_p.size()
        _, c_m, _, _, _ = x_m.size()

        x_p = self._preprocess(x_p, c_p, self.data_bn_p)
        x_m = self._preprocess(x_m, c_m, self.data_bn_m)

        pose_stage_feats = []
        motion_stage_feats = []
        contrastive_pose_feats = []
        contrastive_motion_feats = []
        mid_fusion_weights = {}

        for stage_idx, (pose_stage, motion_stage) in enumerate(
            zip(self.pose_stages, self.motion_stages),
            start=1,
        ):
            x_p = pose_stage(x_p)
            x_m = motion_stage(x_m)

            if str(stage_idx) in self.fusions:
                x_p, x_m, weights = self.fusions[str(stage_idx)](x_p, x_m)
                mid_fusion_weights[str(stage_idx)] = weights

            if self.use_layer_contrastive and label is not None and self.training:
                contrastive_pose_feats.append(x_p)
                contrastive_motion_feats.append(x_m)

            if get_hidden_feat:
                pose_stage_feats.append(x_p)
                motion_stage_feats.append(x_m)

        fused_feat = 0.5 * (x_p + x_m)
        pose_repr = self._global_pool(x_p, n, m, self.pose_pool)
        motion_repr = self._global_pool(x_m, n, m, self.motion_pool)
        fused_repr = self._global_pool(fused_feat, n, m, self.fusion_pool)
        emotion_aux = None
        if self.emotion_condition is not None:
            pose_repr, motion_repr, fused_repr, emotion_aux = self.emotion_condition(
                pose_repr,
                motion_repr,
                fused_repr,
                emotion_feat=emotion_feat,
            )

        logits, aff_pred, aux = self.dla_head(pose_repr, motion_repr, fused_repr)
        aux.update(
            {
                "pose": pose_repr,
                "motion": motion_repr,
                "fusion": fused_repr,
                "mamba_backend": self.mamba_backend,
                "mamba_mtm_kernel_sizes": self.mamba_mtm_kernel_sizes,
                "use_emotion_feature": self.use_emotion_feature,
                "use_part_gate": self.use_part_gate,
                "use_st_attention_pool": self.use_st_attention_pool,
                "mid_fusion_weights": mid_fusion_weights,
            }
        )
        if emotion_aux is not None:
            aux.update(emotion_aux)
        if self.use_layer_contrastive and label is not None and self.training:
            if m != self.contrastive_n_person:
                raise ValueError(
                    "contrastive_n_person={} but input has M={}.".format(
                        self.contrastive_n_person,
                        m,
                    )
                )
            pose_cl_terms = []
            motion_cl_terms = []
            pose_cl_logit = aux.get("pose_logits", logits).detach()
            motion_cl_logit = aux.get("motion_logits", logits).detach()
            cl_label = label.detach()
            for stage_idx, (pose_feat, motion_feat, pose_head, motion_head, stage_weight) in enumerate(
                zip(
                    contrastive_pose_feats,
                    contrastive_motion_feats,
                    self.pose_contrastive_heads,
                    self.motion_contrastive_heads,
                    self.contrastive_stage_weights,
                ),
                start=1,
            ):
                pose_cl = pose_head(pose_feat, cl_label, pose_cl_logit)
                motion_cl = motion_head(motion_feat, cl_label, motion_cl_logit)
                pose_stage_cl = pose_cl * float(stage_weight)
                motion_stage_cl = motion_cl * float(stage_weight)
                pose_cl_terms.append(pose_stage_cl)
                motion_cl_terms.append(motion_stage_cl)
                aux["contrastive_pose_stage_{}".format(stage_idx)] = pose_stage_cl.detach()
                aux["contrastive_motion_stage_{}".format(stage_idx)] = motion_stage_cl.detach()
            if pose_cl_terms:
                pose_cl_loss = torch.stack(pose_cl_terms).sum()
                motion_cl_loss = torch.stack(motion_cl_terms).sum()
                contrastive_loss = self.contrastive_weight * (pose_cl_loss + motion_cl_loss)
                aux["contrastive_loss_pose"] = pose_cl_loss
                aux["contrastive_loss_motion"] = motion_cl_loss
                aux["contrastive_loss"] = contrastive_loss
                aux["contrastive_weight"] = logits.new_tensor(self.contrastive_weight)
        if get_hidden_feat:
            aux.update(
                {
                    "pose_stage_feats": pose_stage_feats,
                    "motion_stage_feats": motion_stage_feats,
                }
            )
        if collect_visuals:
            aux["visuals"] = {
                "pose_dndt": [],
                "motion_dndt": [],
                "csta": {},
                "mid_fusion": mid_fusion_weights,
            }

        return logits, aff_pred, aux


__all__ = [
    "GMamba",
    "HGMBlock",
    "AdaptiveGCN",
    "CrossStreamsGate",
    "MTM",
    "BiMambaBlock",
    "EmoConditioning",
    "PartAwareGate",
    "SpatialTemporalAttentionPool",
]

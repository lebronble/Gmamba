import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveLossFusion(nn.Module):
    """Fuse classification and affective losses without changing the backbone outputs."""

    def __init__(
        self,
        aff_loss_type="MSE",
        aff_loss_ramp_epochs=10,
        loss_min_cls_weight=0.6,
        loss_min_aff_weight=0.05,
        loss_entropy_weight=0.01,
        class_loss_weights=None,
        class_loss_smoothing=0.0,
        use_dynamic_weights=True,
    ):
        super().__init__()
        self.aff_loss_type = str(aff_loss_type)
        self.aff_loss_ramp_epochs = int(aff_loss_ramp_epochs)
        self.loss_min_cls_weight = float(loss_min_cls_weight)
        self.loss_min_aff_weight = float(loss_min_aff_weight)
        self.loss_entropy_weight = float(loss_entropy_weight)
        self.class_loss_smoothing = float(class_loss_smoothing)
        self.use_dynamic_weights = bool(use_dynamic_weights)
        if self.class_loss_smoothing < 0 or self.class_loss_smoothing >= 1:
            raise ValueError("class_loss_smoothing must be in [0, 1).")
        if class_loss_weights is None:
            class_loss_weights = []
        self.register_buffer(
            "class_loss_weights",
            torch.as_tensor(class_loss_weights, dtype=torch.float32),
        )

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

    def forward(
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

        weights = aux.get("loss_weights", None) if self.use_dynamic_weights else None
        has_dynamic_weights = weights is not None
        if not has_dynamic_weights:
            weights = logits.new_full((logits.size(0), 2), 0.5)
        else:
            if weights.dim() != 2 or weights.size(0) != logits.size(0) or weights.size(1) != 2:
                raise ValueError("loss_weights must have shape [N, 2], got {}.".format(tuple(weights.shape)))
            weights = weights.to(dtype=logits.dtype, device=logits.device)
            weights = weights.clamp_min(1e-6)
            weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-6)

        min_cls = float(self.loss_min_cls_weight)
        min_aff = float(self.loss_min_aff_weight)
        if min_cls < 0 or min_aff < 0 or min_cls + min_aff >= 1:
            raise ValueError("loss_min_cls_weight + loss_min_aff_weight must be in [0, 1).")
        min_weights = logits.new_tensor([min_cls, min_aff]).view(1, 2)
        weights = min_weights + (1.0 - min_cls - min_aff) * weights

        aff_scale = self.get_aff_loss_scale(epoch)
        loss_vec = weights[:, 0] * cls_vec + weights[:, 1] * (aff_scale * aff_vec)
        loss = loss_vec.mean()
        if has_dynamic_weights and self.loss_entropy_weight > 0:
            entropy = -(weights * torch.log(weights + 1e-8)).sum(dim=1).mean()
            entropy_penalty = math.log(2.0) - entropy
            loss = loss + self.loss_entropy_weight * entropy_penalty

        metrics = {
            "loss_cls": cls_vec.mean(),
            "loss_aff": aff_vec.mean(),
            "loss_weight_cls": weights[:, 0].mean(),
            "loss_weight_aff": weights[:, 1].mean(),
            "aff_scale": logits.new_tensor(aff_scale),
        }
        return logits, loss, metrics


class SimpleLossFusion(AdaptiveLossFusion):
    """Fixed-ratio classification and affective loss fusion."""

    def __init__(self, *args, **kwargs):
        kwargs["use_dynamic_weights"] = False
        super().__init__(*args, **kwargs)


class AdaptiveTripleCELossFusion(nn.Module):
    """Adaptively fuse fusion, skeleton and emotion cross-entropy losses."""

    def __init__(
        self,
        aff_loss_type="CE",
        aff_loss_ramp_epochs=10,
        loss_min_cls_weight=0.6,
        loss_min_aff_weight=0.05,
        loss_entropy_weight=0.0,
        class_loss_weights=None,
        class_loss_smoothing=0.0,
    ):
        super().__init__()
        self.aff_loss_type = str(aff_loss_type)
        self.aff_loss_ramp_epochs = int(aff_loss_ramp_epochs)
        self.loss_min_cls_weight = float(loss_min_cls_weight)
        self.loss_min_aff_weight = float(loss_min_aff_weight)
        self.loss_entropy_weight = float(loss_entropy_weight)
        self.class_loss_smoothing = float(class_loss_smoothing)
        if self.class_loss_smoothing < 0 or self.class_loss_smoothing >= 1:
            raise ValueError("class_loss_smoothing must be in [0, 1).")
        if class_loss_weights is None:
            class_loss_weights = []
        self.register_buffer(
            "class_loss_weights",
            torch.as_tensor(class_loss_weights, dtype=torch.float32),
        )
        self.loss_logits = nn.Parameter(torch.zeros(3))

    @staticmethod
    def parse_model_output(output):
        return AdaptiveLossFusion.parse_model_output(output)

    def get_aff_loss_scale(self, epoch):
        ramp_epochs = int(self.aff_loss_ramp_epochs)
        if ramp_epochs <= 0:
            return 1.0
        return min(1.0, float(epoch + 1) / ramp_epochs)

    def _class_weights(self, logits):
        class_weights = self.class_loss_weights
        if class_weights.numel() == 0:
            return None
        if class_weights.numel() != logits.size(1):
            raise ValueError(
                "class_loss_weights has {} values, but logits has {} classes.".format(
                    class_weights.numel(),
                    logits.size(1),
                )
            )
        return class_weights.to(device=logits.device, dtype=logits.dtype)

    def _ce_vec(self, logits, label):
        return F.cross_entropy(
            logits,
            label,
            weight=self._class_weights(logits),
            label_smoothing=self.class_loss_smoothing,
            reduction="none",
        )

    def _loss_weights(self, logits):
        min_cls = float(self.loss_min_cls_weight)
        min_aux = float(self.loss_min_aff_weight)
        if min_cls < 0 or min_aux < 0 or min_cls + 2.0 * min_aux >= 1:
            raise ValueError("loss_min_cls_weight + 2 * loss_min_aff_weight must be in [0, 1).")
        base = torch.softmax(self.loss_logits.to(device=logits.device, dtype=logits.dtype), dim=0)
        min_weights = logits.new_tensor([min_cls, min_aux, min_aux])
        weights = min_weights + (1.0 - min_cls - 2.0 * min_aux) * base
        return weights.view(1, 3).expand(logits.size(0), -1)

    def forward(
        self,
        output,
        label,
        feature,
        epoch,
        training=False,
        shuffle_affective_target=False,
    ):
        logits, aff_pred, aux = self.parse_model_output(output)
        ce_logits = aux.get("ce_logits", {})
        fusion_logits = ce_logits.get("fusion", logits)
        skeleton_logits = ce_logits.get("skeleton", None)
        emotion_logits = ce_logits.get("emotion", None)
        if skeleton_logits is None or emotion_logits is None:
            raise ValueError("AdaptiveTripleCELossFusion requires aux['ce_logits']['skeleton'] and ['emotion'].")

        fusion_vec = self._ce_vec(fusion_logits, label)
        skeleton_vec = self._ce_vec(skeleton_logits, label)
        emotion_vec = self._ce_vec(emotion_logits, label)

        weights = self._loss_weights(fusion_logits)
        aux_scale = self.get_aff_loss_scale(epoch)
        loss_vec = (
            weights[:, 0] * fusion_vec
            + aux_scale * weights[:, 1] * skeleton_vec
            + aux_scale * weights[:, 2] * emotion_vec
        )
        loss = loss_vec.mean()
        if self.loss_entropy_weight > 0:
            entropy = -(weights * torch.log(weights + 1e-8)).sum(dim=1).mean()
            entropy_penalty = math.log(3.0) - entropy
            loss = loss + self.loss_entropy_weight * entropy_penalty

        aux_vec = 0.5 * (skeleton_vec + emotion_vec)
        metrics = {
            "loss_cls": fusion_vec.mean(),
            "loss_aff": aux_vec.mean(),
            "loss_skeleton": skeleton_vec.mean(),
            "loss_emotion": emotion_vec.mean(),
            "loss_weight_cls": weights[:, 0].mean(),
            "loss_weight_aff": (weights[:, 1] + weights[:, 2]).mean(),
            "loss_weight_skeleton": weights[:, 1].mean(),
            "loss_weight_emotion": weights[:, 2].mean(),
            "aff_scale": logits.new_tensor(aux_scale),
        }
        return fusion_logits, loss, metrics


__all__ = ["AdaptiveLossFusion", "SimpleLossFusion", "AdaptiveTripleCELossFusion"]

# ------------------------------------------------------------------------
# Plain-DETR
# Copyright (c) 2023 Xi'an Jiaotong University & Microsoft Research Asia.
# Licensed under The MIT License [see LICENSE for details]
# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Backbone modules.
"""
import math
import os
import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List

from util.misc import NestedTensor, is_main_process

from .position_encoding import build_position_encoding
from .swin_transformer_v2 import SwinTransformerV2
from .utils import LayerNorm2D
from .dinov3_backbone import build_dinov3_backbone


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n, eps=1e-5):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))
        self.eps = eps

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        num_batches_tracked_key = prefix + "num_batches_tracked"
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = self.eps
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):
    def __init__(
        self, backbone: nn.Module, train_backbone: bool, return_interm_layers: bool
    ):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if (
                not train_backbone
                or "layer2" not in name
                and "layer3" not in name
                and "layer4" not in name
            ):
                parameter.requires_grad_(False)
        if return_interm_layers:
            # return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
            return_layers = {"layer2": "0", "layer3": "1", "layer4": "2"}
            self.strides = [8, 16, 32]
            self.num_channels = [512, 1024, 2048]
        else:
            return_layers = {"layer4": "0"}
            self.strides = [32]
            self.num_channels = [2048]
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)

    def forward(self, tensor_list: NestedTensor):
        xs = self.body(tensor_list.tensors)
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""

    def __init__(
        self,
        name: str,
        train_backbone: bool,
        return_interm_layers: bool,
        dilation: bool,
    ):
        norm_layer = FrozenBatchNorm2d
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=is_main_process(),
            norm_layer=norm_layer,
        )
        assert name not in ("resnet18", "resnet34"), "number of channels are hard coded"
        super().__init__(backbone, train_backbone, return_interm_layers)
        if dilation:
            self.strides[-1] = self.strides[-1] // 2


class TransformerBackbone(nn.Module):
    def __init__(
        self, backbone: str, train_backbone: bool, return_interm_layers: bool, args
    ):
        super().__init__()
        out_indices = (1, 2, 3) if return_interm_layers else (3,)

        if backbone == "swin_v2_small_window16":
            backbone = SwinTransformerV2(pretrain_img_size=256,
                                         embed_dim=96,
                                         depths=[2, 2, 18, 2],
                                         num_heads=[3, 6, 12, 24],
                                         window_size=16,
                                         drop_path_rate=args.drop_path_rate,
                                         use_checkpoint=args.use_checkpoint,
                                         out_indices=out_indices,
                                         pretrained_window_size=[16, 16, 16, 8],
                                         global_blocks=[[-1], [-1], [-1], [-1]])
            embed_dim = 96
            backbone.init_weights(args.pretrained_backbone_path)
        elif backbone == "swin_v2_small_window16_2global":
            backbone = SwinTransformerV2(pretrain_img_size=256,
                                                embed_dim=96,
                                                depths=[2, 2, 18, 2],
                                                num_heads=[3, 6, 12, 24],
                                                window_size=16,
                                                drop_path_rate=args.drop_path_rate,
                                                use_checkpoint=args.use_checkpoint,
                                                out_indices=out_indices,
                                                pretrained_window_size=[16, 16, 16, 8],
                                                global_blocks=[[-1], [-1], [-1], [0, 1]])
            embed_dim = 96
            backbone.init_weights(args.pretrained_backbone_path)
        elif backbone == "swin_v2_small_window12to16":
            backbone = SwinTransformerV2(pretrain_img_size=256,
                                         embed_dim=96,
                                         depths=[2, 2, 18, 2],
                                         num_heads=[3, 6, 12, 24],
                                         window_size=16,
                                         drop_path_rate=args.drop_path_rate,
                                         use_checkpoint=args.use_checkpoint,
                                         out_indices=out_indices,
                                         pretrained_window_size=[12, 12, 12, 6],
                                         global_blocks=[[-1], [-1], [-1], [-1]])
            embed_dim = 96
            backbone.init_weights(args.pretrained_backbone_path)
        elif backbone == "swin_v2_small_window12to16_2global":
            backbone = SwinTransformerV2(pretrain_img_size=256,
                                                embed_dim=96,
                                                depths=[2, 2, 18, 2],
                                                num_heads=[3, 6, 12, 24],
                                                window_size=16,
                                                drop_path_rate=args.drop_path_rate,
                                                use_checkpoint=args.use_checkpoint,
                                                out_indices=out_indices,
                                                pretrained_window_size=[12, 12, 12, 6],
                                                global_blocks=[[-1], [-1], [-1], [0, 1]])
            embed_dim = 96
            backbone.init_weights(args.pretrained_backbone_path)
        else:
            raise NotImplementedError

        for name, parameter in backbone.named_parameters():
            if not train_backbone:
                parameter.requires_grad_(False)

        if return_interm_layers:
            self.strides = [8, 16, 32]
            self.num_channels = [
                embed_dim * 2,
                embed_dim * 4,
                embed_dim * 8,
            ]
        else:
            self.strides = [32]
            self.num_channels = [embed_dim * 8]

        self.body = backbone

    def forward(self, tensor_list: NestedTensor):
        xs = self.body(tensor_list.tensors)

        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out


class UpSampleWrapper(nn.Module):
    """Upsample last feat map to specific stride."""

    def __init__(
        self,
        net,
        stride,
    ):
        """
        Args:
            net (Backbone): module representing the subnetwork backbone.
                Must be a subclass of :class:`Backbone`.
            dim (int): number of channels of fpn hidden features.
        """
        super(UpSampleWrapper, self).__init__()

        self.net = net

        assert len(net.strides) == 1, 'UpSample should receive one input.'
        in_stride = net.strides[0]
        self.strides = [stride]

        assert len(net.num_channels) == 1, 'UpSample should receive one input.'
        in_num_channel = net.num_channels[0]

        assert stride <= in_stride, 'Target stride is larger than input stride.'
        if stride == in_stride:
            self.upsample = nn.Identity()
            self.num_channels = net.num_channels
        else:
            scale = int(math.log2(in_stride // stride))
            dim = in_num_channel
            layers = []
            for _ in range(scale-1):
                layers += [
                    nn.ConvTranspose2d(dim, dim // 2, kernel_size=2, stride=2),
                    LayerNorm2D(dim // 2),
                    nn.GELU()
                ]
                dim = dim // 2
            layers += [nn.ConvTranspose2d(dim, dim // 2, kernel_size=2, stride=2)]
            dim = dim // 2
            self.upsample = nn.Sequential(*layers)
            self.num_channels = [dim]

    def forward(self, tensor_list: NestedTensor):
        xs = self.net(tensor_list)

        assert len(xs) == 1

        out: Dict[str, NestedTensor] = {}
        for name, value in xs.items():
            m = tensor_list.mask
            assert m is not None
            x = self.upsample(value.tensors)
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out

class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)
        self.strides = backbone.strides
        self.num_channels = backbone.num_channels

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in sorted(xs.items()):
            out.append(x)

        # position encoding
        for idx, x in enumerate(out):
            pos.append(self[1][idx](x).to(x.tensors.dtype))

        return out, pos


def _unwrap_state_dict(checkpoint):
    if not isinstance(checkpoint, dict):
        raise RuntimeError("Invalid checkpoint format")
    for k in ["model", "state_dict", "teacher", "student", "backbone"]:
        if k in checkpoint and isinstance(checkpoint[k], dict):
            return checkpoint[k]
    return checkpoint


def _maybe_strip_prefix(state_dict, prefix):
    if not state_dict:
        return state_dict
    keys = list(state_dict.keys())
    if not all(k.startswith(prefix) for k in keys[: min(len(keys), 100)]):
        return state_dict
    return {k[len(prefix) :]: v for k, v in state_dict.items()}


def _load_dinov3_backbone_model(args):
    if args.dinov3_repo is None or args.dinov3_model is None:
        raise ValueError("dinov3_repo and dinov3_model must be set when using a dinov3 backbone")

    if not os.path.exists(args.dinov3_repo):
        raise ValueError(f"dinov3_repo path does not exist: {args.dinov3_repo}")

    backbone_model = torch.hub.load(args.dinov3_repo, args.dinov3_model, source="local", pretrained=False)

    if args.dinov3_weights:
        checkpoint = torch.load(args.dinov3_weights, map_location="cpu")
        state_dict = _unwrap_state_dict(checkpoint)
        for prefix in ["module.", "backbone.", "teacher.", "student.", "model."]:
            state_dict = _maybe_strip_prefix(state_dict, prefix)
        msg = backbone_model.load_state_dict(state_dict, strict=False)
        print(msg)

    return backbone_model


def build_backbone(args):
    position_embedding = build_position_encoding(args)
    train_backbone = args.lr_backbone > 0
    return_interm_layers = args.masks or (args.num_feature_levels > 1)

    if args.backbone.startswith("dinov3"):
        backbone_model = _load_dinov3_backbone_model(args)
        return build_dinov3_backbone(backbone_model, args)
    if "resnet" in args.backbone:
        backbone = Backbone(
            args.backbone, train_backbone, return_interm_layers, args.dilation,
        )
    else:
        backbone = TransformerBackbone(
            args.backbone, train_backbone, return_interm_layers, args
        )

    if args.upsample_backbone_output:
        backbone = UpSampleWrapper(
            backbone,
            args.upsample_stride,
        )

    model = Joiner(backbone, position_embedding)
    return model
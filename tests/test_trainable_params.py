import argparse

import torch

from models.dinov3_backbone import build_dinov3_backbone

from tests.fake_dinov3 import FakeDinoV3Model


def _make_args(lr_backbone=0.0, layers_to_use=None, n_windows_sqrt=0, backbone_use_layernorm=False, blocks_to_train=None):
    return argparse.Namespace(
        hidden_dim=256,
        num_feature_levels=1,
        position_embedding="sine",
        lr_backbone=lr_backbone,
        layers_to_use=layers_to_use,
        n_windows_sqrt=n_windows_sqrt,
        backbone_use_layernorm=backbone_use_layernorm,
        blocks_to_train=blocks_to_train,
    )


def _count_trainable(module):
    n = 0
    for p in module.parameters():
        if p.requires_grad:
            n += p.numel()
    return n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--unfreeze", action="store_true", default=False)
    args_cli = parser.parse_args()

    lr_backbone = 1e-5 if args_cli.unfreeze else 0.0

    fake = FakeDinoV3Model(patch_size=16, n_blocks=48, embed_dim=256)
    args = _make_args(
        lr_backbone=lr_backbone,
        layers_to_use=[10, 20, 30, 40],
        n_windows_sqrt=0,
        backbone_use_layernorm=True,
        blocks_to_train=None,
    )

    backbone = build_dinov3_backbone(fake, args)

    trainable = _count_trainable(backbone)
    print("trainable_params", trainable)

    for name, p in backbone.named_parameters():
        if p.requires_grad:
            print("trainable", name, p.numel())

    if not args_cli.unfreeze:
        assert trainable == 0

    print("OK")


if __name__ == "__main__":
    main()

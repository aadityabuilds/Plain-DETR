import argparse

import torch

from util.misc import NestedTensor
from models.dinov3_backbone import build_dinov3_backbone

from tests.fake_dinov3 import FakeDinoV3Model


def _make_args(
    hidden_dim=256,
    num_feature_levels=1,
    position_embedding="sine",
    lr_backbone=0.0,
    layers_to_use=None,
    n_windows_sqrt=0,
    backbone_use_layernorm=False,
    blocks_to_train=None,
):
    return argparse.Namespace(
        hidden_dim=hidden_dim,
        num_feature_levels=num_feature_levels,
        position_embedding=position_embedding,
        lr_backbone=lr_backbone,
        layers_to_use=layers_to_use,
        n_windows_sqrt=n_windows_sqrt,
        backbone_use_layernorm=backbone_use_layernorm,
        blocks_to_train=blocks_to_train,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h", type=int, default=1536)
    parser.add_argument("--w", type=int, default=2304)
    parser.add_argument("--patch", type=int, default=16)
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--layers", type=int, nargs="+", default=[10, 20, 30, 40])
    parser.add_argument("--windows", type=int, default=0)
    args_cli = parser.parse_args()

    fake = FakeDinoV3Model(patch_size=args_cli.patch, n_blocks=48, embed_dim=args_cli.embed_dim)
    args = _make_args(
        hidden_dim=256,
        num_feature_levels=1,
        lr_backbone=0.0,
        layers_to_use=args_cli.layers,
        n_windows_sqrt=args_cli.windows,
        backbone_use_layernorm=False,
    )

    backbone = build_dinov3_backbone(fake, args)

    b = 1
    x = torch.zeros((b, 3, args_cli.h, args_cli.w), dtype=torch.float32)
    mask = torch.zeros((b, args_cli.h, args_cli.w), dtype=torch.bool)
    nt = NestedTensor(x, mask)

    feats, pos = backbone(nt)

    assert len(feats) == 1
    assert len(pos) == 1

    f0 = feats[0].tensors
    hh = args_cli.h // args_cli.patch
    ww = args_cli.w // args_cli.patch
    assert f0.shape[-2:] == (hh, ww)

    expected_c = len(args_cli.layers) * args_cli.embed_dim
    if args_cli.windows and args_cli.windows > 0:
        expected_c = expected_c * 2

    assert f0.shape[1] == expected_c

    print("OK")
    print("feature", tuple(f0.shape))
    print("expected_c", expected_c)


if __name__ == "__main__":
    main()

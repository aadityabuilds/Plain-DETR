import argparse

import torch

from models import build_model


def _parse_args():
    from main import get_args_parser

    parser = get_args_parser()
    args = parser.parse_args([])

    args.device = "cpu"
    args.dataset_file = "coco"
    args.num_classes = 1

    args.backbone = "resnet50"
    args.num_feature_levels = 1

    args.hidden_dim = 256
    args.dim_feedforward = 2048
    args.nheads = 8
    args.dropout = 0.0

    args.decoder_type = "global_rpe_decomp"
    args.decoder_rpe_type = "linear"
    args.decoder_rpe_hidden_dim = 512
    args.proposal_feature_levels = 1
    args.proposal_in_stride = 16
    args.proposal_tgt_strides = [16]

    args.two_stage = True
    args.mixed_selection = True

    args.num_queries_one2one = 16
    args.num_queries_one2many = 0

    args.with_box_refine = True
    args.aux_loss = True

    args.add_transformer_encoder = True
    args.num_encoder_layers = 6
    args.norm_type = "pre_norm"

    args.lr_backbone = 0.0

    return args


def main():
    args = _parse_args()
    model, _, _ = build_model(args)
    model.eval()

    b = 2
    h = 128
    w = 128
    x = torch.randn((b, 3, h, w), dtype=torch.float32)

    with torch.no_grad():
        out = model([x[i] for i in range(b)])

    assert "pred_logits" in out
    assert "pred_boxes" in out

    num_classes = args.num_classes
    assert out["pred_logits"].shape == (b, args.num_queries_one2one, num_classes)
    assert out["pred_boxes"].shape == (b, args.num_queries_one2one, 4)

    assert out["pred_logits_one2many"].shape == (b, args.num_queries_one2many, num_classes)
    assert out["pred_boxes_one2many"].shape == (b, args.num_queries_one2many, 4)

    if args.aux_loss:
        assert "aux_outputs" in out
        assert isinstance(out["aux_outputs"], list)
    else:
        assert "aux_outputs" not in out

    print("OK")
    print("pred_logits", tuple(out["pred_logits"].shape))
    print("pred_boxes", tuple(out["pred_boxes"].shape))


if __name__ == "__main__":
    main()

import torch

from models import build_model


def _make_args(add_encoder: bool):
    from main import get_args_parser

    parser = get_args_parser()
    args = parser.parse_args([])

    args.device = "cpu"
    args.dataset_file = "coco"
    args.num_classes = 1

    args.backbone = "resnet50"
    args.num_feature_levels = 1

    args.hidden_dim = 128
    args.dim_feedforward = 256
    args.nheads = 4
    args.dropout = 0.0

    args.decoder_type = "global_rpe_decomp"
    args.decoder_rpe_type = "linear"
    args.decoder_rpe_hidden_dim = 128
    args.proposal_feature_levels = 1
    args.proposal_in_stride = 16
    args.proposal_tgt_strides = [16]

    args.two_stage = True
    args.mixed_selection = True

    args.num_queries_one2one = 4
    args.num_queries_one2many = 0

    args.with_box_refine = True
    args.aux_loss = False

    args.add_transformer_encoder = add_encoder
    args.num_encoder_layers = 2
    args.norm_type = "pre_norm"

    args.lr_backbone = 0.0

    return args


def _run_once(add_encoder: bool):
    args = _make_args(add_encoder)
    model, _, _ = build_model(args)
    model.eval()

    calls = {"n": 0}

    enc = model.transformer.encoder
    if enc is not None:
        def hook(_m, _inp, _out):
            calls["n"] += 1
        enc.register_forward_hook(hook)

    b = 2
    x = torch.randn((b, 3, 64, 64), dtype=torch.float32)
    with torch.no_grad():
        _ = model([x[i] for i in range(b)])

    return calls["n"], enc is not None


def main():
    n_on, has_enc_on = _run_once(True)
    n_off, has_enc_off = _run_once(False)

    assert has_enc_on
    assert n_on == 1
    assert (not has_enc_off) or n_off == 0

    print("OK")
    print("encoder_on_calls", n_on)
    print("encoder_off_calls", n_off)


if __name__ == "__main__":
    main()

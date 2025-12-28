import torch
from torch import nn


class FakeDinoV3Model(nn.Module):
    def __init__(self, patch_size=16, n_blocks=48, embed_dim=256, embed_dims=None):
        super().__init__()
        self.patch_size = patch_size
        self.n_blocks = n_blocks
        self.embed_dim = embed_dim
        if embed_dims is None:
            self.embed_dims = [embed_dim for _ in range(n_blocks)]
        else:
            if len(embed_dims) != n_blocks:
                raise ValueError("embed_dims length must equal n_blocks")
            self.embed_dims = list(embed_dims)

        self._dummy = nn.Parameter(torch.zeros(1), requires_grad=False)

    def get_intermediate_layers(self, x, n=1, reshape=True):
        if not reshape:
            raise ValueError("FakeDinoV3Model only supports reshape=True")
        if isinstance(n, int):
            blocks = list(range(self.n_blocks - n, self.n_blocks))
        else:
            blocks = list(n)
        b, _, h, w = x.shape
        hh = h // self.patch_size
        ww = w // self.patch_size
        outs = []
        for idx in blocks:
            c = self.embed_dims[idx]
            outs.append(torch.zeros((b, c, hh, ww), dtype=x.dtype, device=x.device))
        return outs

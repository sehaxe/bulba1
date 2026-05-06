import torch
import torch.nn as nn
import torch.nn.functional as F
from bulba1.model.bit_linear import BitLinear, ste_b158


class Mamba2SSD(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.d_model = cfg.d_model
        self.d_state = cfg.mamba_d_state
        self.d_conv = cfg.mamba_d_conv
        self.expand = cfg.mamba_expand
        self.d_inner = int(self.expand * cfg.d_model)
        use_bit = getattr(cfg, "use_bitlinear", False)

        Linear = BitLinear if use_bit else nn.Linear
        self.in_proj = Linear(cfg.d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=self.d_conv,
            groups=self.d_inner,
            padding=self.d_conv - 1,
            bias=True,
        )
        self.x_proj = Linear(self.d_inner, self.d_state * 2 + 1, bias=False)
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)
        self.A_log = nn.Parameter(
            torch.log(torch.arange(1, self.d_state + 1)).repeat(self.d_inner, 1)
        )
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = Linear(self.d_inner, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        xz = self.in_proj(x)
        x_inner, z = xz.chunk(2, dim=-1)

        x_inner = x_inner.transpose(1, 2)
        x_inner = self.conv1d(x_inner)[..., :T]
        x_inner = x_inner.transpose(1, 2)
        x_inner = F.silu(x_inner)

        x_db = self.x_proj(x_inner)
        dt, B_ssm, C_ssm = torch.split(x_db, [1, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt))

        A = -torch.exp(self.A_log.to(x.dtype))
        y = _ssd_scan_compiled(x_inner, dt, A, B_ssm, C_ssm, self.D)

        y = y * F.silu(z)
        return self.out_proj(y)

    def _ssd_scan(self, x, dt, A, B, C):
        B_batch, T, D = x.shape
        dt = dt.squeeze(-1)
        dA = torch.exp(A.unsqueeze(0).unsqueeze(0) * dt.unsqueeze(-1))
        dB = dt.unsqueeze(-1) * B.unsqueeze(2)

        h = torch.zeros(B_batch, D, self.d_state, device=x.device, dtype=x.dtype)
        ys = []
        for t in range(T):
            h = dA[:, t] * h + dB[:, t] * x[:, t].unsqueeze(-1)
            y = torch.sum(h * C[:, t].unsqueeze(1), dim=-1)
            ys.append(y)
        y = torch.stack(ys, dim=1)
        return y + self.D.view(1, 1, -1) * x


def _parallel_scan_affine_4d(a, B):
    B_batch, T, D, S = B.shape
    if T == 1:
        return B
    a_cum = a.clone()
    B_cum = B.clone()
    step = 1
    while step < T:
        a_left = torch.cat(
            [
                torch.ones(B_batch, step, D, S, device=a.device, dtype=a.dtype),
                a_cum[:, :-step],
            ],
            dim=1,
        )
        B_left = torch.cat(
            [
                torch.zeros(B_batch, step, D, S, device=B.device, dtype=B.dtype),
                B_cum[:, :-step],
            ],
            dim=1,
        )
        a_old = a_cum
        B_old = B_cum
        a_cum = a_left * a_old
        B_cum = a_old * B_left + B_old
        step *= 2
    return B_cum


def _ssd_scan_compiled(x, dt, A, B, C, D):
    B_batch, T, D_dim = x.shape
    dt = dt.squeeze(-1)
    dA = torch.exp(A.unsqueeze(0).unsqueeze(0) * dt.unsqueeze(-1))
    dB = dt.unsqueeze(-1) * B.unsqueeze(2)
    B_mat = dB * x.unsqueeze(-1)
    h = _parallel_scan_affine_4d(dA, B_mat)
    y = torch.sum(h * C.unsqueeze(2), dim=-1)
    return (y + D.view(1, 1, -1) * x).clone()

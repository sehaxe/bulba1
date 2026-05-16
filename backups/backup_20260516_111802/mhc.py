import torch
import torch.nn as nn
import torch.nn.functional as F

class MHC(nn.Module):
    """
    Усиленный MHC (DeepSeek) с защитами от взрыва активаций.
    - n=4 (полноценный residual stream)
    - H_res включён (Sinkhorn-Knopp)
    - Дополнительные clamp'ы и нормализация
    - Совместим с autocast (все операции в том же dtype)
    """
    def __init__(self, cfg):
        super().__init__()
        self.d_model = cfg.d_model
        self.n = getattr(cfg, "mhc_n", 4)
        self.iterations = getattr(cfg, "mhc_iterations", 10)

        # Только H_pre и H_post – H_res не создаётся, он вычисляется на лету
        self.phi_pre = nn.Linear(self.d_model, self.n, bias=False)
        self.phi_post = nn.Linear(self.d_model, self.n, bias=False)

        self.alpha_pre = nn.Parameter(torch.ones(1) * 0.01)
        self.alpha_post = nn.Parameter(torch.ones(1) * 0.01)
        self.b_pre = nn.Parameter(torch.zeros(1, self.n))
        self.b_post = nn.Parameter(torch.zeros(1, self.n))

        # Параметры для H_res (смешивание потоков)
        self.phi_res = nn.Linear(self.d_model, self.n * self.n, bias=False)
        self.alpha_res = nn.Parameter(torch.ones(1) * 0.01)
        self.b_res = nn.Parameter(torch.zeros(self.n, self.n))

    def sinkhorn_knopp(self, M, iterations):
        for _ in range(iterations):
            M = M / (M.sum(dim=-2, keepdim=True) + 1e-8)
            M = M / (M.sum(dim=-1, keepdim=True) + 1e-8)
        return M

    def forward(self, x, residual_fn, *args, past_kv=None, **kwargs):
        B, T, C = x.shape
        n = self.n
        dtype = x.dtype

        # 1. Расширяем residual stream до n копий
        x_expanded = x.unsqueeze(2).expand(-1, -1, n, -1)   # (B,T,n,C)

        # 2. Агрегированное представление (среднее по потокам)
        x_flat = x_expanded.mean(dim=2)                      # (B,T,C)

        # 3. H_pre – агрегация потоков для входа подслоя
        h_pre_raw = self.alpha_pre * self.phi_pre(x_flat) + self.b_pre
        H_pre = torch.softmax(h_pre_raw, dim=-1)             # выпуклая комбинация
        H_pre = H_pre.clamp(min=1e-4, max=1.0)               # защита от вырождения

        # 4. H_post – модуляция выхода подслоя
        h_post_raw = self.alpha_post * self.phi_post(x_flat) + self.b_post
        H_post = 0.5 * torch.tanh(h_post_raw)                # диапазон [-0.5, 0.5]
        H_post = H_post.clamp(min=-0.8, max=0.8)             # дополнительная защита

        # 5. H_res – перемешивание потоков (дважды стохастическая матрица)
        h_res_raw = self.alpha_res * self.phi_res(x_flat) + self.b_res.view(1, 1, n * n)
        h_res_raw = h_res_raw.view(B, T, n, n)
        H_res = self.sinkhorn_knopp(torch.exp(h_res_raw), self.iterations)
        H_res = H_res.clamp(min=1e-4, max=1.0)               # дважды стохастическая не должна выходить за [0,1]

        # 6. Вход для подслоя
        x_pre = (x_expanded * H_pre.unsqueeze(-1)).sum(dim=2)   # (B,T,C)

        # 7. Вызываем подслой
        try:
            output = residual_fn(x_pre, past_kv=past_kv)
        except TypeError:
            output = residual_fn(x_pre)
        if isinstance(output, tuple):
            main_out = output[0]
            extras = output[1:]
        else:
            main_out = output
            extras = ()

        # 8. Применяем H_res к выходу подслоя
        main_out_expanded = main_out.unsqueeze(2).expand(-1, -1, n, -1)  # (B,T,n,C)
        main_out_mixed = torch.matmul(H_res, main_out_expanded)          # (B,T,n,n)@(B,T,n,C)->(B,T,n,C)
        main_out_mixed = main_out_mixed * H_post.unsqueeze(-1)           # модуляция
        main_out_mixed = main_out_mixed.sum(dim=2)                       # (B,T,C)

        # 9. Применяем H_res к исходному x (остаточная связь через смешивание)
        x_res = torch.matmul(H_res, x_expanded)                         # (B,T,n,C)
        x_res = x_res.sum(dim=2)                                        # (B,T,C)

        # 10. Нормализация: делим на n, т.к. суммирование n копий даёт усиление ~n
        x_res = x_res / n
        main_out_mixed = main_out_mixed / n

        # 11. Остаточная связь
        new_x = x + x_res + main_out_mixed

        # 12. Мягкое ограничение нормы (не даём вырасти больше чем на 30%)
        x_norm = x.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        new_norm = new_x.norm(dim=-1, keepdim=True)
        scale = (x_norm / new_norm).clamp(max=1.3)
        new_x = new_x * scale

        if extras:
            return (new_x,) + extras
        return new_x
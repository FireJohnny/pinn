#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
04_burgers.py —— 复现论文的 Burgers 算例(Appendix A.1 / B.8)

方程:u_t + u u_x - (0.01/π) u_xx = 0,  x∈[-1,1], t∈[0,1]
初值:u(0,x) = -sin(π x),  边界:u(t,-1) = u(t,1) = 0
论文报告:正问题相对 L2 误差 1.97e-3;反问题从 2000 点复现 λ1=1.0, λ2=0.01/π

做法:
  ① 用半隐式有限差分(中心差分 + 隐式扩散)生成高精度真值
  ② 连续时间 PINN 解正问题,与 ① 比相对 L2
  ③ 反问题:把 (λ1, λ2) 设为可训练参数,从 2000 个点(含 1% 噪声)反推
"""
import time, json, math, os
import numpy as np
import torch, torch.nn as nn

torch.set_num_threads(1)
NU = 0.01 / math.pi
DEV = "cpu"
OUT = "/root/ai4s_papers/pinn"


# ═══════════════ ① 高精度真值:半隐式有限差分 ═══════════════
def solve_fd(nx=1001, nt=1000):
    x = np.linspace(-1, 1, nx); dx = x[1] - x[0]
    dt = 1.0 / nt
    u = -np.sin(np.pi * x)
    u[0] = u[-1] = 0.0
    U = np.zeros((nt + 1, nx)); U[0] = u
    for k in range(nt):
        # 显式处理对流(中心差分),隐式处理扩散
        ux = np.zeros_like(u)
        ux[1:-1] = (u[2:] - u[:-2]) / (2 * dx)
        ustar = u - dt * u * ux
        ustar[0] = ustar[-1] = 0.0
        # 隐式扩散: (I - r*L) u^{n+1} = ustar
        r = NU * dt / dx ** 2
        from scipy.linalg import solve_banded
        n = nx - 2
        ab = np.zeros((3, n)); ab[0, 1:] = -r; ab[1, :] = 1 + 2 * r; ab[2, :-1] = -r
        un = solve_banded((1, 1), ab, ustar[1:-1])
        u = np.zeros_like(u); u[1:-1] = un
        U[k + 1] = u
    return x, np.linspace(0, 1, nt + 1), U


class MLP(nn.Module):
    def __init__(self, w=32, d=5):
        super().__init__()
        layers, inp = [], 2
        for _ in range(d):
            layers += [nn.Linear(inp, w), nn.Tanh()]
            inp = w
        layers += [nn.Linear(inp, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x, t):
        return self.net(torch.cat([x, t], dim=1))


def grad(y, x):
    return torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y), create_graph=True)[0]


def train(lambdas=None, obs=None, epochs=4000, n_col=3000, lr=1e-3, w=32, d=5, seed=0):
    """lambdas=None → 已知 (1, 0.01/π);给 (l1,l2) 初值时作为可训练参数反推"""
    torch.manual_seed(seed)
    net = MLP(w, d)
    if lambdas is None:
        par = None; lam = (1.0, NU)
    else:
        par = nn.Parameter(torch.tensor([float(lambdas[0]), float(lambdas[1])]))
        lam = None
    opt = torch.optim.Adam(list(net.parameters()) + ([par] if par is not None else []), lr=lr)

    xc = torch.rand(n_col, 1) * 2 - 1; tc = torch.rand(n_col, 1)
    xb = torch.cat([-torch.ones(200, 1), torch.ones(200, 1)]); tb = torch.rand(400, 1)
    xi = torch.rand(300, 1) * 2 - 1; ti = torch.zeros(300, 1)
    ui = -torch.sin(math.pi * xi)
    if obs is not None:
        xo = torch.tensor(obs[0], dtype=torch.float32).reshape(-1, 1)
        to = torch.tensor(obs[1], dtype=torch.float32).reshape(-1, 1)
        uo = torch.tensor(obs[2], dtype=torch.float32).reshape(-1, 1)

    t0 = time.time()
    for ep in range(epochs):
        opt.zero_grad()
        xc.requires_grad_(True); tc.requires_grad_(True)
        u = net(xc, tc)
        ut = grad(u, tc); ux = grad(u, xc); uxx = grad(ux, xc)
        l1, l2 = (par[0], par[1]) if par is not None else lam
        loss_pde = ((ut + l1 * u * ux - l2 * uxx) ** 2).mean()
        loss_bc = (net(xb, tb) ** 2).mean()
        loss = loss_pde + 100 * loss_bc
        if obs is None:
            loss = loss + 100 * ((net(xi, ti) - ui) ** 2).mean()
        else:
            loss = loss + 100 * ((net(xo, to) - uo) ** 2).mean()
        loss.backward(); opt.step()
        if ep % 1000 == 0 or ep == epochs - 1:
            extra = f" λ1={par[0].item():.4f} λ2={par[1].item():.5f}" if par is not None else ""
            print(f"  ep {ep:>4} loss {loss.item():.2e} pde {loss_pde.item():.2e}{extra} | {time.time()-t0:.0f}s", flush=True)
    return net, par, time.time() - t0


def rel_l2(net, x, t, U):
    """PINN 与 FD 真值的相对 L2 误差(在 FD 网格上算)"""
    GX, GT = np.meshgrid(x, t[::10])
    UU = U[::10]
    with torch.no_grad():
        p = net(torch.tensor(GX.reshape(-1, 1), dtype=torch.float32),
                torch.tensor(GT.reshape(-1, 1), dtype=torch.float32)).numpy().reshape(GX.shape)
    return np.linalg.norm(p - UU) / np.linalg.norm(UU), p, GX, GT, UU


if __name__ == "__main__":
    print("① 生成高精度真值(半隐式有限差分)")
    t0 = time.time()
    x, t, U = solve_fd()
    print(f"  网格 {U.shape[1]}×{U.shape[0]} | 耗时 {time.time()-t0:.1f}s "
          f"| max|u| = {np.abs(U).max():.3f}", flush=True)
    np.savez_compressed(f"{OUT}/burgers_truth.npz", x=x, t=t, U=U)

    res = {}
    print("\n② 正问题:PINN vs 有限差分")
    net, _, sec = train(epochs=4000)
    err, pred, GX, GT, UU = rel_l2(net, x, t, U)
    print(f"  相对 L2 = {err:.3e} | 耗时 {sec:.0f}s | 论文报告 1.97e-3")
    res["forward"] = dict(rel_l2=float(err), sec=sec, paper=1.97e-3)
    torch.save(net.state_dict(), f"{OUT}/burgers_fwd.pt")
    np.savez_compressed(f"{OUT}/burgers_pred.npz", pred=pred, GX=GX, GT=GT, UU=UU)

    print("\n③ 反问题:2000 点 + 1% 噪声,反推 λ1、λ2(真值 1.0 与 0.0031831)")
    rng = np.random.default_rng(0)
    n_obs = 2000
    xo = rng.uniform(-1, 1, n_obs); to = rng.uniform(0, 1, n_obs)
    # 用 FD 真值插值出观测
    from scipy.interpolate import RegularGridInterpolator
    f = RegularGridInterpolator((t, x), U, bounds_error=False, fill_value=0.0)
    uo = f(np.column_stack([to, xo]))
    uo_n = uo * (1 + 0.01 * rng.standard_normal(n_obs))
    net_i, par, sec_i = train(lambdas=(0.8, 0.002), obs=(xo, to, uo_n), epochs=4000)
    l1, l2 = float(par[0]), float(par[1])
    err_i, *_ = rel_l2(net_i, x, t, U)
    print(f"  反推 λ1 = {l1:.4f}(真值 1.0,误差 {abs(l1-1)*100:.2f}%)")
    print(f"  反推 λ2 = {l2:.5f}(真值 {NU:.5f},误差 {abs(l2-NU)/NU*100:.2f}%)")
    print(f"  解相对 L2 = {err_i:.3e} | 耗时 {sec_i:.0f}s")
    res["inverse"] = dict(l1=l1, l2=l2, l1_err=abs(l1 - 1) * 100,
                          l2_err=abs(l2 - NU) / NU * 100,
                          rel_l2=float(err_i), sec=sec_i, n_obs=n_obs)
    json.dump(res, open(f"{OUT}/burgers_results.json", "w"), ensure_ascii=False, indent=1)
    print("\n" + "=" * 58)
    print(f"正问题:本文 {err:.2e} vs 论文 1.97e-03")
    print(f"反问题:λ1 误差 {abs(l1-1)*100:.2f}%,λ2 误差 {abs(l2-NU)/NU*100:.2f}%")
    print("=" * 58)

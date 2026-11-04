#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
02_compare.py —— PINN 的两个面:正问题 vs 传统数值解;反问题反推参数

① 正问题:同一道 1D 热传导方程,比较
   - 有限差分(隐式/Crank-Nicolson,经典数值解)
   - PINN(Raissi 2019 的做法)
   两者都与解析解比,并记录耗时。

② 反问题:只给少量带噪观测,PINN 把扩散系数 α 当作可训练参数反推出来
   —— 这是有限差分做不到的事(经典数值解要求先知道 α)
"""
import time, math, json
import numpy as np
import torch
import torch.nn as nn
from scipy.linalg import solve_banded

torch.set_num_threads(1)
ALPHA_TRUE = 0.1
NX, NT = 101, 101


def analytic(x, t):
    return np.exp(-ALPHA_TRUE * math.pi ** 2 * t) * np.sin(math.pi * x)


# ---------------- ① 有限差分(Crank-Nicolson) ----------------
def solve_fd(nx=NX, nt=NT, alpha=ALPHA_TRUE):
    t0 = time.time()
    x = np.linspace(0, 1, nx); dx = x[1] - x[0]
    t = np.linspace(0, 1, nt); dt = t[1] - t[0]
    r = alpha * dt / (2 * dx ** 2)
    n = nx - 2
    # 三对角矩阵 (I - r*L) u^{n+1} = (I + r*L) u^n,Dirichlet 边界为 0
    ab = np.zeros((3, n))
    ab[0, 1:] = -r; ab[1, :] = 1 + 2 * r; ab[2, :-1] = -r
    U = np.zeros((nt, nx))
    U[0, 1:-1] = np.sin(math.pi * x[1:-1])
    for k in range(nt - 1):
        u = U[k, 1:-1]
        rhs = u + r * (np.concatenate([[0], u[:-1]]) - 2 * u + np.concatenate([u[1:], [0]]))
        U[k + 1, 1:-1] = solve_banded((1, 1), ab, rhs)
    return U, time.time() - t0


# ---------------- PINN 构建模块 ----------------
class MLP(nn.Module):
    def __init__(self, w=32, d=3):
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


def train_pinn(alpha=None, obs=None, epochs=4000, n_col=2000, lr=1e-3, seed=0):
    """alpha=None 视为已知常数;alpha 为 float 时作为可训练参数反推。
    obs = (x, t, u) 观测数据,给定时加入数据损失(反问题)。"""
    torch.manual_seed(seed)
    net = MLP()
    if alpha is None:
        a_param = None; a_val = ALPHA_TRUE
    else:
        a_param = nn.Parameter(torch.tensor([float(alpha)]))
        a_val = None
    params = list(net.parameters()) + ([a_param] if a_param is not None else [])
    opt = torch.optim.Adam(params, lr=lr)

    xc = torch.rand(n_col, 1); tc = torch.rand(n_col, 1)
    xb = torch.cat([torch.zeros(200, 1), torch.ones(200, 1)]); tb = torch.rand(400, 1)
    xi = torch.rand(200, 1); ti = torch.zeros(200, 1)
    ui = torch.sin(math.pi * xi)
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
        a = a_param if a_param is not None else a_val
        loss_pde = ((ut - a * uxx) ** 2).mean()
        loss_bc = (net(xb, tb) ** 2).mean()
        loss = loss_pde + 100 * loss_bc
        if obs is None:
            loss = loss + 100 * ((net(xi, ti) - ui) ** 2).mean()
        else:
            loss = loss + 100 * ((net(xo, to) - uo) ** 2).mean()
        loss.backward(); opt.step()
        if ep % 1000 == 0 or ep == epochs - 1:
            extra = f" α={a_param.item():.4f}" if a_param is not None else ""
            print(f"  ep {ep:>4} loss {loss.item():.2e} pde {loss_pde.item():.2e}{extra} | {time.time()-t0:.0f}s", flush=True)
    return net, a_param, time.time() - t0


def grid_err(net):
    g = np.linspace(0, 1, NX)
    GX, GT = np.meshgrid(g, g)
    with torch.no_grad():
        pred = net(torch.tensor(GX.reshape(-1, 1), dtype=torch.float32),
                   torch.tensor(GT.reshape(-1, 1), dtype=torch.float32)).numpy().reshape(GX.shape)
    true = analytic(GX, GT)
    return pred, true, np.abs(pred - true), np.linalg.norm(pred - true) / np.linalg.norm(true) * 100


if __name__ == "__main__":
    res = {}

    print("─" * 60)
    print("① 有限差分(Crank-Nicolson)")
    U_fd, t_fd = solve_fd()
    x = np.linspace(0, 1, NX); t = np.linspace(0, 1, NT)
    GX, GT = np.meshgrid(x, t)
    truth = analytic(GX, GT)
    e_fd = np.abs(U_fd - truth)
    rel_fd = np.linalg.norm(U_fd - truth) / np.linalg.norm(truth) * 100
    print(f"  耗时 {t_fd:.3f}s | 最大绝对误差 {e_fd.max():.2e} | 相对误差 {rel_fd:.4f}%")
    res["fd"] = dict(sec=t_fd, max_abs=float(e_fd.max()), rel=float(rel_fd))

    print("─" * 60)
    print("② PINN 正问题(α 已知)")
    net_f, _, t_pinn = train_pinn(epochs=4000)
    pred_p, truth_p, e_p, rel_p = grid_err(net_f)
    print(f"  耗时 {t_pinn:.1f}s | 最大绝对误差 {e_p.max():.4f} | 相对误差 {rel_p:.3f}%")
    res["pinn_fwd"] = dict(sec=t_pinn, max_abs=float(e_p.max()), rel=float(rel_p))

    print("─" * 60)
    print("③ PINN 反问题:只用 30 个带 1% 噪声的观测点反推 α(真值 0.1)")
    rng = np.random.default_rng(0)
    n_obs = 30
    xo = rng.uniform(0, 1, n_obs); to = rng.uniform(0, 1, n_obs)
    uo = analytic(xo, to) * (1 + 0.01 * rng.standard_normal(n_obs))
    net_i, a_param, t_inv = train_pinn(alpha=0.05, obs=(xo, to, uo), epochs=4000)
    a_hat = float(a_param.item())
    pred_i, truth_i, e_i, rel_i = grid_err(net_i)
    print(f"  反推 α = {a_hat:.4f}(真值 0.1000,相对误差 {abs(a_hat-0.1)/0.1*100:.1f}%)")
    print(f"  耗时 {t_inv:.1f}s | 解的最大绝对误差 {e_i.max():.4f} | 相对误差 {rel_i:.3f}%")
    res["pinn_inv"] = dict(sec=t_inv, alpha_hat=a_hat, alpha_err=abs(a_hat - 0.1) / 0.1 * 100,
                           max_abs=float(e_i.max()), rel=float(rel_i), n_obs=n_obs)

    json.dump(res, open("/root/ai4s_papers/pinn/results.json", "w"), ensure_ascii=False, indent=1)
    print("\n" + "=" * 60)
    print(f"{'方法':<18}{'耗时':>10}{'相对误差':>12}")
    print(f"{'有限差分':<18}{t_fd:>9.3f}s{rel_fd:>11.4f}%")
    print(f"{'PINN(正问题)':<17}{t_pinn:>9.1f}s{rel_p:>11.3f}%")
    print(f"{'PINN(反问题)':<17}{t_inv:>9.1f}s{rel_i:>11.3f}%  α={a_hat:.4f}")
    print("=" * 60)
    torch.save(net_f.state_dict(), "/root/ai4s_papers/pinn/forward.pt")
    torch.save(net_i.state_dict(), "/root/ai4s_papers/pinn/inverse.pt")

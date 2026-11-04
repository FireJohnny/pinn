#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
06_diagnose.py —— 判断"是数据不行还是复现有问题"

① 真值体检:两套分辨率交叉对比 + 守恒量检查
   —— 若两套网格一致,说明 FD 真值可信,问题不在数据
② 复现体检:只改一个变量(损失权重),看误差是否骤降
   —— 论文用 MSE_u + MSE_f(不加权);我之前给初值/边界乘了 100
"""
import time, json, math, sys
import numpy as np
import torch, torch.nn as nn
from scipy.linalg import solve_banded
from scipy.stats import qmc

torch.set_num_threads(1)
NU = 0.01 / math.pi
OUT = "/root/ai4s_papers/pinn"


def solve_fd(nx=1001, nt=1000):
    x = np.linspace(-1, 1, nx); dx = x[1] - x[0]; dt = 1.0 / nt
    u = -np.sin(np.pi * x); u[0] = u[-1] = 0.0
    U = np.zeros((nt + 1, nx)); U[0] = u
    for k in range(nt):
        ux = np.zeros_like(u); ux[1:-1] = (u[2:] - u[:-2]) / (2 * dx)
        us = u - dt * u * ux; us[0] = us[-1] = 0.0
        r = NU * dt / dx ** 2; n = nx - 2
        ab = np.zeros((3, n)); ab[0, 1:] = -r; ab[1, :] = 1 + 2 * r; ab[2, :-1] = -r
        un = solve_banded((1, 1), ab, us[1:-1])
        u = np.zeros_like(u); u[1:-1] = un
        U[k + 1] = u
    return x, np.linspace(0, 1, nt + 1), U


# ═════════════ ① 真值体检 ═════════════
print("① 真值体检:两套分辨率对比")
t0 = time.time()
x1, t1, U1 = solve_fd(1001, 1000)
x2, t2, U2 = solve_fd(2001, 4000)
# 插值到共同网格比较
from scipy.interpolate import RegularGridInterpolator
f2 = RegularGridInterpolator((t2, x2), U2, bounds_error=False, fill_value=0.0)
sub = t1[::50]
diff = np.array([U1[i] - f2((t1[i] * np.ones_like(x1), x1)) for i in range(0, len(t1), 50)])
rel = np.linalg.norm(diff) / np.linalg.norm(U1[::50])
print(f"  (1001×1000) vs (2001×4000):相对差异 {rel:.3e}")
print(f"  两套网格{'一致 → 真值可信' if rel < 5e-3 else '不一致 → 真值有问题!'}")

# 守恒量:∫u dx 的演变(应为单调衰减,不应反号/爆炸)
mass = np.trapezoid(U1, x1, axis=1)   # numpy 2 已移除 np.trapz
print(f"  ∫u dx:t=0 为 {mass[0]:.4f},t=1 为 {mass[-1]:.4f},单调衰减 = {np.all(np.diff(mass) <= 1e-6)}")
print(f"  极值:min {U1.min():.4f},max {U1.max():.4f}(应在 ±1 内)")
print(f"  耗时 {time.time()-t0:.1f}s\n")


# ═════════════ ② 复现体检:权重 A/B ═════════════
class MLP(nn.Module):
    def __init__(self, w=32, d=5):
        super().__init__()
        layers, inp = [], 2
        for _ in range(d):
            layers += [nn.Linear(inp, w), nn.Tanh()]; inp = w
        layers += [nn.Linear(inp, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x, t):
        return self.net(torch.cat([x, t], dim=1))


def grad(y, x):
    return torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y), create_graph=True)[0]


def run(w_ic=1.0, n_col=3000, w=32, d=5, epochs=4000, lr=1e-3, seed=0):
    torch.manual_seed(seed)
    net = MLP(w, d)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    s = qmc.LatinHypercube(d=2, seed=seed).random(n_col)
    xc = torch.tensor(s[:, 0] * 2 - 1, dtype=torch.float32).reshape(-1, 1)
    tc = torch.tensor(s[:, 1], dtype=torch.float32).reshape(-1, 1)
    xb = torch.cat([-torch.ones(200, 1), torch.ones(200, 1)]); tb = torch.rand(400, 1)
    xi = torch.rand(300, 1) * 2 - 1; ti = torch.zeros(300, 1)
    ui = -torch.sin(math.pi * xi)
    t0 = time.time()
    for ep in range(epochs):
        opt.zero_grad()
        xc.requires_grad_(True); tc.requires_grad_(True)
        u = net(xc, tc)
        ut = grad(u, tc); ux = grad(u, xc); uxx = grad(ux, xc)
        f = ut + u * ux - NU * uxx
        mse_f = (f ** 2).mean()
        xa = torch.cat([xi, xb]); ta = torch.cat([ti, tb])
        ua = torch.cat([ui, torch.zeros(400, 1)])
        mse_u = ((net(xa, ta) - ua) ** 2).mean()
        loss = mse_f + w_ic * mse_u
        loss.backward(); opt.step()
        if ep % 1000 == 0 or ep == epochs - 1:
            print(f"    ep {ep:>4} L {loss.item():.3e} (方程 {mse_f.item():.3e} / 数据 {mse_u.item():.3e}) | {time.time()-t0:.0f}s", flush=True)
    # 误差
    gx, gt = np.meshgrid(x1, t1[::10])
    with torch.no_grad():
        p = net(torch.tensor(gx.reshape(-1, 1), dtype=torch.float32),
                torch.tensor(gt.reshape(-1, 1), dtype=torch.float32)).numpy().reshape(gx.shape)
    UU = U1[::10]
    return np.linalg.norm(p - UU) / np.linalg.norm(UU), net, time.time() - t0


print("② 复现体检:A/B 只改损失权重(其余完全相同:5×32、3000 LHS、4000 轮)")
print("  ── A:加权 100(我上一版的做法)")
e100, net100, s100 = run(w_ic=100.0)
print(f"     相对 L2 = {e100:.3e}(此前复现的 45.8%)\n")
print("  ── B:不加权(论文的做法 MSE_u + MSE_f)")
e1, net1, s1 = run(w_ic=1.0)
print(f"     相对 L2 = {e1:.3e}\n")
print("=" * 62)
print(f"加权 100:{e100:.2%}  |  不加权:{e1:.2%}  |  改善 {e100/max(e1,1e-12):.1f} 倍")
print(f"论文(9层×20、10000 配点):0.197%")
print("=" * 62)
json.dump(dict(truth_rel=float(rel), w100=float(e100), w1=float(e1), paper=1.97e-3),
          open(f"{OUT}/diagnose.json", "w"), indent=1)
np.savez_compressed(f"{OUT}/diagnose_pred.npz", U=U1, x=x1, t=t1)
torch.save(net1.state_dict(), f"{OUT}/burgers_unweighted.pt")

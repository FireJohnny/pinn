#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
08_adaptive.py —— 残差自适应撒点(RAD)能否用同样算力把激波压准

对比(点数与网络完全相同,只改"配点怎么分布"):
  固定 LHS 3000 点(基线 D):8.57%
  RAD:3000 点,每 500 轮按残差大小重采样(候选池 20000 点)

同时量化"点有没有聚到激波上":统计 |x|<0.03 且 t>0.35 区域内的配点数
"""
import time, json, math
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


def residual(net, x, t):
    # 必须显式开梯度:若在 no_grad 上下文中调用,net(x,t) 不会有 grad_fn,autograd.grad 会报错
    with torch.enable_grad():
        x = x.clone().requires_grad_(True); t = t.clone().requires_grad_(True)
        u = net(x, t)
        ut = grad(u, t); ux = grad(u, x); uxx = grad(ux, x)
        f = (ut + u * ux - NU * uxx).detach().abs().flatten()
    return f


def in_layer(x, t):
    """激波所在区域:x≈0 且 t>0.35"""
    return (np.abs(x) < 0.03) & (t > 0.35)


def run(adaptive=True, n_col=3000, n_pool=20000, epochs=6000, resample_every=500,
        w=32, d=5, lr=1e-3, seed=3, k=4.0):
    torch.manual_seed(seed)
    net = MLP(w, d)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.StepLR(opt, step_size=2500, gamma=0.5)

    s = qmc.LatinHypercube(d=2, seed=seed).random(n_pool)
    px = torch.tensor(s[:, 0] * 2 - 1, dtype=torch.float32).reshape(-1, 1)
    pt = torch.tensor(s[:, 1], dtype=torch.float32).reshape(-1, 1)
    # 初始配点:候选池前 n_col 个(与固定基线同一分布起点)
    xc, tc = px[:n_col].clone(), pt[:n_col].clone()

    xb = torch.cat([-torch.ones(200, 1), torch.ones(200, 1)]); tb = torch.rand(400, 1)
    xi = torch.rand(300, 1) * 2 - 1; ti = torch.zeros(300, 1)
    ui = -torch.sin(math.pi * xi)
    xa = torch.cat([xi, xb]); ta = torch.cat([ti, tb]); ua = torch.cat([ui, torch.zeros(400, 1)])

    n_layer_before = int(in_layer(xc.numpy().ravel(), tc.numpy().ravel()).sum())
    t0 = time.time()
    for ep in range(epochs):
        if adaptive and ep > 0 and ep % resample_every == 0:
            f = residual(net, px, pt)   # 内部已 enable_grad
            p = (f ** k).numpy() + 1e-12
            p = p / p.sum()
            rng = np.random.default_rng(seed + ep)
            idx = rng.choice(n_pool, size=n_col, replace=True, p=p)
            # 保留 20% 均匀点,否则全部挤进激波层、其他区域失去覆盖
            nmix = n_col // 5
            idx[:nmix] = rng.choice(n_pool, size=nmix, replace=False)
            xc = px[idx].clone(); tc = pt[idx].clone()
            nl = int(in_layer(xc.numpy().ravel(), tc.numpy().ravel()).sum())
            print(f"    重采样 @{ep}:层内配点 {nl}/{n_col}(初 {n_layer_before})"
                  f" | 残差最大 {f.max():.2e} | {time.time()-t0:.0f}s", flush=True)
        opt.zero_grad()
        xx = xc.clone().requires_grad_(True); tt = tc.clone().requires_grad_(True)
        u = net(xx, tt)
        ut = grad(u, tt); ux = grad(u, xx); uxx = grad(ux, xx)
        mse_f = ((ut + u * ux - NU * uxx) ** 2).mean()
        mse_u = ((net(xa, ta) - ua) ** 2).mean()
        (mse_f + mse_u).backward(); opt.step(); sch.step()
        if ep % 1000 == 0 or ep == epochs - 1:
            nl = int(in_layer(xc.numpy().ravel(), tc.numpy().ravel()).sum())
            print(f"  ep {ep:>4} L {(mse_f+mse_u).item():.3e} (方程 {mse_f.item():.3e} / 数据 {mse_u.item():.3e})"
                  f" 层内点 {nl} | {time.time()-t0:.0f}s", flush=True)

    # L-BFGS 精修(用最终这套点)
    lbfgs = torch.optim.LBFGS(net.parameters(), lr=1.0, max_iter=1500,
                              history_size=50, line_search_fn="strong_wolfe")
    def closure():
        lbfgs.zero_grad()
        xx = xc.clone().requires_grad_(True); tt = tc.clone().requires_grad_(True)
        u = net(xx, tt)
        ut = grad(u, tt); ux = grad(u, xx); uxx = grad(ux, xx)
        l = ((ut + u * ux - NU * uxx) ** 2).mean() + ((net(xa, ta) - ua) ** 2).mean()
        l.backward(); return l
    lbfgs.step(closure)
    print(f"  L-BFGS 后 loss {closure().item():.3e} | {time.time()-t0:.0f}s", flush=True)

    gx, gt = np.meshgrid(x_truth, t_truth[::10])
    with torch.no_grad():
        p = net(torch.tensor(gx.reshape(-1, 1), dtype=torch.float32),
                torch.tensor(gt.reshape(-1, 1), dtype=torch.float32)).numpy().reshape(gx.shape)
    UU = U_truth[::10]
    err = np.linalg.norm(p - UU) / np.linalg.norm(UU)
    return err, p, gx, gt, UU, time.time() - t0, net, xc.numpy().ravel(), tc.numpy().ravel()


if __name__ == "__main__":
    x_truth, t_truth, U_truth = solve_fd()
    print("残差自适应撒点(RAD):3000 点、候选池 20000、每 500 轮重采样、k=4、保留 20% 均匀点", flush=True)
    err, p, gx, gt, UU, sec, net, xc, tc = run(adaptive=True)
    nl = int(in_layer(xc, tc).sum())
    print("\n" + "=" * 64)
    print(f"RAD(自适应):相对 L2 = {err*100:.3f}%  | 层内配点 {nl}/3000 | 耗时 {sec:.0f}s")
    print(f"固定 LHS(基线 D):8.570%  | 层内配点 ~36/3000")
    print(f"论文:0.197%")
    print("=" * 64)
    json.dump(dict(rad_rel=float(err), sec=sec, n_layer=int(nl),
                   fixed_rel=0.0857, paper=1.97e-3),
              open(f"{OUT}/adaptive.json", "w"), indent=1)
    np.savez_compressed(f"{OUT}/adaptive_pred.npz", pred=p, GX=gx, GT=gt, UU=UU,
                        xc=xc, tc=tc)
    torch.save(net.state_dict(), f"{OUT}/burgers_rad.pt")

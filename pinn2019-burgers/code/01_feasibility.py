#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
01_feasibility.py —— PINN 在本机(2 核 / 1.6G / 单线程 torch)是否可行

方程:一维热传导 u_t = α u_xx,x∈[0,1], t∈[0,1]
初值 u(x,0) = sin(πx),边界 u(0,t)=u(1,t)=0,α=0.1
解析解 u = exp(-α π² t) sin(πx)  ← 可与 PINN 逐点比误差
"""
import time, math
import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(1)
ALPHA = 0.1
DEV = "cpu"


def analytic(x, t):
    return np.exp(-ALPHA * math.pi ** 2 * t) * np.sin(math.pi * x)


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


def train(n_col=2000, n_bc=200, n_ic=200, epochs=3000, lr=1e-3, seed=0):
    torch.manual_seed(seed)
    net = MLP()
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    xc = torch.rand(n_col, 1); tc = torch.rand(n_col, 1)
    xb = torch.cat([torch.zeros(n_bc, 1), torch.ones(n_bc, 1)]); tb = torch.rand(2 * n_bc, 1)
    xi = torch.rand(n_ic, 1); ti = torch.zeros(n_ic, 1)
    ui = torch.sin(math.pi * xi)

    t0 = time.time()
    for ep in range(epochs):
        opt.zero_grad()
        xc.requires_grad_(True); tc.requires_grad_(True)
        u = net(xc, tc)
        u_t = grad(u, tc); u_x = grad(u, xc)
        u_xx = grad(u_x, xc)
        r = u_t - ALPHA * u_xx
        loss_pde = (r ** 2).mean()
        loss_bc = (net(xb, tb) ** 2).mean()
        loss_ic = ((net(xi, ti) - ui) ** 2).mean()
        loss = loss_pde + 100 * loss_bc + 100 * loss_ic
        loss.backward(); opt.step()
        if ep % 500 == 0 or ep == epochs - 1:
            print(f"  ep {ep:>4} loss {loss.item():.3e} pde {loss_pde.item():.3e} "
                  f"ic {loss_ic.item():.3e} | {time.time()-t0:.0f}s", flush=True)
    return net, time.time() - t0


if __name__ == "__main__":
    net, sec = train(epochs=3000)
    # 与解析解比误差
    g = np.linspace(0, 1, 101)
    GX, GT = np.meshgrid(g, g)
    with torch.no_grad():
        pred = net(torch.tensor(GX.reshape(-1, 1), dtype=torch.float32),
                   torch.tensor(GT.reshape(-1, 1), dtype=torch.float32)).numpy().reshape(GX.shape)
    true = analytic(GX, GT)
    err = np.abs(pred - true)
    print(f"\n训练耗时 {sec:.1f}s")
    print(f"最大绝对误差 {err.max():.4f} | 平均 {err.mean():.4f} | 相对误差(对真值范数) "
          f"{np.linalg.norm(pred-true)/np.linalg.norm(true)*100:.2f}%")
    torch.save(net.state_dict(), "/root/ai4s_papers/pinn/feasibility.pt")

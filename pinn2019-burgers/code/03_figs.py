#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""03_figs.py —— PINN 文章配图(框架示意 / 正问题 / 方法对比 / 反问题)"""
import os, re, json, math
import numpy as np
import torch, torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from scipy.linalg import solve_banded

FONT = "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"
fm.fontManager.addfont(FONT)
plt.rcParams["font.family"] = "WenQuanYi Micro Hei"
plt.rcParams["axes.unicode_minus"] = False

BASE = "/root/ai4s_papers/pinn"
FIG = "/root/ai4s_papers/figs"
os.makedirs(FIG, exist_ok=True)
torch.set_num_threads(1)
ALPHA_TRUE = 0.1
NX, NT = 101, 101

res = json.load(open(f"{BASE}/results.json"))


def analytic(x, t):
    return np.exp(-ALPHA_TRUE * math.pi ** 2 * t) * np.sin(math.pi * x)


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


def predict(path):
    net = MLP(); net.load_state_dict(torch.load(path)); net.eval()
    g = np.linspace(0, 1, NX)
    GX, GT = np.meshgrid(g, g)
    with torch.no_grad():
        p = net(torch.tensor(GX.reshape(-1, 1), dtype=torch.float32),
                torch.tensor(GT.reshape(-1, 1), dtype=torch.float32)).numpy().reshape(GX.shape)
    return GX, GT, p


def solve_fd(nx=NX, nt=NT, alpha=ALPHA_TRUE):
    x = np.linspace(0, 1, nx); dx = x[1] - x[0]
    t = np.linspace(0, 1, nt); dt = t[1] - t[0]
    r = alpha * dt / (2 * dx ** 2); n = nx - 2
    ab = np.zeros((3, n)); ab[0, 1:] = -r; ab[1, :] = 1 + 2 * r; ab[2, :-1] = -r
    U = np.zeros((nt, nx)); U[0, 1:-1] = np.sin(math.pi * x[1:-1])
    for k in range(nt - 1):
        u = U[k, 1:-1]
        rhs = u + r * (np.concatenate([[0], u[:-1]]) - 2 * u + np.concatenate([u[1:], [0]]))
        U[k + 1, 1:-1] = solve_banded((1, 1), ab, rhs)
    return U


# ══════════════ 图1:PINN 框架示意 ══════════════
fig, ax = plt.subplots(figsize=(11, 4.4), dpi=200)
ax.set_xlim(0, 11); ax.set_ylim(0, 4.4); ax.axis("off")


def box(x, y, w, h, txt, fc="#eef4fb", ec="#3b6ea5", fs=11, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12",
                                fc=fc, ec=ec, lw=1.6))
    ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", linespacing=1.5)


def arrow(x1, y1, x2, y2, color="#555", style="->"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                 mutation_scale=14, color=color, lw=1.5))


box(0.25, 2.55, 1.5, 0.95, "输入\n$(x,\\,t)$", fc="#fdf2e9", ec="#c87f0a")
box(2.15, 2.3, 2.2, 1.45, "全连接网络\n3 层 × 32\nTanh 激活", fc="#eef4fb", bold=True)
box(4.75, 2.55, 1.45, 0.95, "输出\n$u(x,t)$", fc="#e8f6ee", ec="#1e7d43")
arrow(1.8, 3.02, 2.12, 3.02)
arrow(4.4, 3.02, 4.72, 3.02)

# 损失三项
box(6.9, 3.10, 3.8, 0.72, "① 方程残差  $\\mathcal{L}_{PDE}=\\Vert u_t-\\alpha u_{xx}\\Vert^2$", fc="#fdecea", ec="#c0392b", fs=10.5)
box(6.9, 2.10, 3.8, 0.72, "② 初值 + 边界  $\\mathcal{L}_{IC}+\\mathcal{L}_{BC}$", fc="#fdecea", ec="#c0392b", fs=10.5)
box(6.9, 1.10, 3.8, 0.72, "③ 观测数据  $\\mathcal{L}_{data}$(反问题才需要)", fc="#fdecea", ec="#c0392b", fs=10.5)
arrow(6.22, 3.02, 6.86, 3.46, color="#3b6ea5")
arrow(6.22, 3.02, 6.86, 2.46, color="#3b6ea5")
arrow(6.22, 3.02, 6.86, 1.46, color="#3b6ea5")

ax.text(5.5, 0.35, "关键:方程不是用来算的,而是变成损失函数里的一项 —— 网络在「满足物理」的前提下拟合数据",
        ha="center", fontsize=11, color="#c0392b")
plt.tight_layout()
plt.savefig(f"{FIG}/fig_pinn_framework.png", bbox_inches="tight", facecolor="white")
plt.close()
print("✅ fig_pinn_framework.png")

# ══════════════ 图2:正问题对比 ══════════════
GX, GT, pred = predict(f"{BASE}/forward.pt")
truth = analytic(GX, GT)
U_fd = solve_fd()
err = np.abs(pred - truth)
fig, axes = plt.subplots(1, 3, figsize=(14, 4.0), dpi=200)
for ax, M, ttl, kw in [
    (axes[0], pred, f"PINN 解(相对误差 {res['pinn_fwd']['rel']:.2f}%)", dict(cmap="viridis")),
    (axes[1], truth, "解析解(真值)", dict(cmap="viridis")),
    (axes[2], err, f"绝对误差(最大 {err.max():.4f})", dict(cmap="Reds"))]:
    im = ax.pcolormesh(GX, GT, M, shading="auto", **kw)
    ax.set_xlabel("位置 $x$"); ax.set_title(ttl, fontsize=11.5)
    plt.colorbar(im, ax=ax, fraction=0.046)
axes[0].set_ylabel("时间 $t$")
plt.tight_layout()
plt.savefig(f"{FIG}/fig_pinn_forward.png", bbox_inches="tight", facecolor="white")
plt.close()
print("✅ fig_pinn_forward.png")

# ══════════════ 图3:方法对比 ══════════════
fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.6), dpi=200,
                         gridspec_kw={"width_ratios": [1, 1.15]})
ax = axes[0]
names = ["有限差分\n(Crank-Nicolson)", "PINN\n(正问题)", "PINN\n(反问题)"]
secs = [res["fd"]["sec"], res["pinn_fwd"]["sec"], res["pinn_inv"]["sec"]]
rels = [res["fd"]["rel"], res["pinn_fwd"]["rel"], res["pinn_inv"]["rel"]]
cols = ["#1e7d43", "#c0392b", "#c87f0a"]
b1 = ax.bar(names, rels, color=cols, width=0.55)
for b, v, s in zip(b1, rels, secs):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.4f}%\n{s:.3f}s" if s < 1 else f"{v:.3f}%\n{s:.0f}s",
            ha="center", fontsize=10, fontweight="bold")
ax.set_ylabel("相对误差(vs 解析解,%)", fontsize=11.5)
ax.set_title("① 精度:有限差分比 PINN 高两个数量级", fontsize=12.5, pad=10)
ax.set_ylim(0, max(rels) * 1.45)
ax.grid(axis="y", alpha=0.25, ls=":")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)

ax = axes[1]
# 用柱状而不是散点:三个方法在精度与耗时上都跨 5 个数量级,散点会让标注全挤在角落
b2 = ax.bar(names, secs, color=cols, width=0.55)
ax.set_yscale("log")
for b, s in zip(b2, secs):
    txt = f"{s:.3f} s" if s < 1 else f"{s:.0f} s"
    ax.text(b.get_x() + b.get_width() / 2, s * 1.35, txt, ha="center",
            fontsize=10.5, fontweight="bold")
ax.set_ylabel("耗时(秒,对数轴)", fontsize=11.5)
ax.set_title("② 代价:同一道题,有限差分快约 3.6 万倍", fontsize=12.5, pad=10)
ax.set_ylim(2e-3, 2000)
ax.grid(axis="y", alpha=0.25, ls=":", which="major")
for s_ in ("top", "right"):
    ax.spines[s_].set_visible(False)
ax.text(0.02, 0.94, "同一台机器、同一道方程、同一套解析解对标",
        transform=ax.transAxes, fontsize=9.5, color="#555", va="top")

plt.savefig(f"{FIG}/fig_pinn_benchmark.png", bbox_inches="tight", facecolor="white")
plt.close()
print("✅ fig_pinn_benchmark.png")

# ══════════════ 图4:反问题 ══════════════
rng = np.random.default_rng(0)
n_obs = 30
xo = rng.uniform(0, 1, n_obs); to = rng.uniform(0, 1, n_obs)
fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.6), dpi=200)
ax = axes[0]
im = ax.pcolormesh(GX, GT, truth, shading="auto", cmap="Blues", alpha=0.75)
ax.scatter(xo, to, s=46, c="#c0392b", edgecolor="white", lw=1.0, zorder=3,
           label=f"{n_obs} 个观测点(带 1% 噪声)")
ax.set_xlabel("位置 $x$"); ax.set_ylabel("时间 $t$")
ax.set_title("只给这些零星观测,要求同时反推 α 和解", fontsize=12, pad=10)
ax.legend(loc="upper right", fontsize=10, framealpha=0.92)
plt.colorbar(im, ax=ax, fraction=0.046, label="u")

ax = axes[1]
log = open("/root/pinn_compare.log").read()
seg = log.split("③")[-1]
eps, alphas = [], []
for m in re.finditer(r"ep\s+(\d+) .*?α=([\d.]+)", seg):
    eps.append(int(m.group(1))); alphas.append(float(m.group(2)))
ax.plot(eps, alphas, "o-", color="#c87f0a", lw=2.4, ms=9, mec="white", mew=1.4, zorder=3)
ax.axhline(ALPHA_TRUE, color="#1e7d43", ls="--", lw=1.8, label=f"真值 α = {ALPHA_TRUE}")
for e, a in zip(eps, alphas):
    ax.annotate(f"{a:.4f}", (e, a), textcoords="offset points", xytext=(0, -18 if a < 0.09 else 12),
                ha="center", fontsize=9.5)
ax.set_xlabel("训练轮次", fontsize=11.5)
ax.set_ylabel("网络学到的 α", fontsize=11.5)
ax.set_title(f"α 从错猜的 0.05 收敛到 {alphas[-1]:.4f}(误差 1.3%)", fontsize=12, pad=10)
ax.legend(fontsize=10.5, loc="lower right", framealpha=0.92)
ax.grid(alpha=0.25, ls=":")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
plt.tight_layout()
plt.savefig(f"{FIG}/fig_pinn_inverse.png", bbox_inches="tight", facecolor="white")
plt.close()
print("✅ fig_pinn_inverse.png")
print("   α 轨迹:", list(zip(eps, alphas)))

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""05_figs_burgers.py —— Burgers 算例配图(真值/预测/误差 + λ 收敛)"""
import os, re, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

FONT = "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"
fm.fontManager.addfont(FONT)
plt.rcParams["font.family"] = "WenQuanYi Micro Hei"
plt.rcParams["axes.unicode_minus"] = False

BASE = "/root/ai4s_papers/pinn"
FIG = "/root/ai4s_papers/figs"
res = json.load(open(f"{BASE}/burgers_results.json"))
D = np.load(f"{BASE}/burgers_pred.npz")
pred, GX, GT, UU = D["pred"], D["GX"], D["GT"], D["UU"]
err = np.abs(pred - UU)

fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.0), dpi=200)
for ax, M, ttl, kw in [
    (axes[0], UU, "有限差分真值(参考解)", dict(cmap="viridis")),
    (axes[1], pred, f"PINN 预测(相对 L2 = {res['forward']['rel_l2']:.1%})", dict(cmap="viridis")),
    (axes[2], err, f"绝对误差(最大 {err.max():.2f})", dict(cmap="Reds"))]:
    im = ax.pcolormesh(GX, GT, M, shading="auto", **kw)
    ax.set_xlabel("位置 $x$")
    ax.set_title(ttl, fontsize=11.5)
    plt.colorbar(im, ax=ax, fraction=0.046)
axes[0].set_ylabel("时间 $t$")
plt.tight_layout()
plt.savefig(f"{FIG}/fig_burgers_forward.png", bbox_inches="tight", facecolor="white")
plt.close()
print("✅ fig_burgers_forward.png")

# λ 收敛
log = open("/root/burgers.log").read()
seg = log.split("③")[-1]
eps, l1s, l2s = [], [], []
for m in re.finditer(r"ep\s+(\d+) .*?λ1=([\d.]+) λ2=([\d.]+)", seg):
    eps.append(int(m.group(1))); l1s.append(float(m.group(2))); l2s.append(float(m.group(3)))

fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.4), dpi=200)
for ax, ys, true, name, unit in [
    (axes[0], l1s, 1.0, "$\\lambda_1$", ""),
    (axes[1], l2s, 0.01 / np.pi, "$\\lambda_2$", "")]:
    ax.plot(eps, ys, "o-", color="#c87f0a", lw=2.4, ms=9, mec="white", mew=1.4, zorder=3)
    ax.axhline(true, color="#1e7d43", ls="--", lw=1.9, label=f"真值 {true:.5f}")
    for e, y in zip(eps, ys):
        ax.annotate(f"{y:.4f}", (e, y), textcoords="offset points", xytext=(0, 11),
                    ha="center", fontsize=9.5)
    ax.set_xlabel("训练轮次", fontsize=11.5)
    ax.set_ylabel(f"网络学到的 {name}", fontsize=11.5)
    ax.set_title(f"{name}:从初值收向真值,最终误差 "
                 f"{abs(ys[-1]-true)/true*100:.1f}%", fontsize=11.5, pad=8)
    ax.legend(fontsize=10.5, loc="best", framealpha=0.92)
    ax.grid(alpha=0.25, ls=":")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
plt.tight_layout()
plt.savefig(f"{FIG}/fig_burgers_inverse.png", bbox_inches="tight", facecolor="white")
plt.close()
print("✅ fig_burgers_inverse.png")
print("   λ1:", l1s, "\n   λ2:", l2s)

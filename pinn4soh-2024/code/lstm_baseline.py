#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
lstm_baseline.py —— 给 PINN4SOH 补一个 LSTM 基线(仓库里没有)

与仓库保持一致的协议:
  · 同一批 MIT 数据(data/MIT data,三个批次,125 节电池)
  · 同一特征(16 个统计特征,CSV 里除 capacity 外全部列)
  · 同一划分(id % 5 == 0 进测试集)
  · 同一归一化(对特征做 min-max 到 [-1,1])
  · SOH = capacity / 1.1 Ah

LSTM 用「最近 L 个循环的特征序列」预测当前循环的 SOH(L=5),
这是这类时序模型的标准用法,也与 MLP/CNN 用的 (x1,x2) 配对形式不同 —— 特此说明。

3 个随机种子(仓库是 10 次,这里为控制 CPU 时间降到 3 次,已在报告中注明)。
"""
import os, glob, math, time, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

torch.set_num_threads(1)
BASE = "/root/ai4s_papers/repro_pinn4soh/PINN4SOH-main/data/MIT data"
OUT = "/root/ai4s_papers/repro_pinn4soh/lstm_result.json"
NOM = 1.1          # MIT A123 额定容量 Ah
L = 5              # 序列长度
EPOCHS = 120
BATCH = 256
SEEDS = [0, 1, 2]


def load_all():
    rows = []
    for batch in ["2017-05-12", "2017-06-30", "2018-04-12"]:
        for f in sorted(glob.glob(f"{BASE}/{batch}/*.csv")):
            bid = int(os.path.basename(f).split("-")[-1].split(".")[0])
            df = pd.read_csv(f)
            if len(df) < L + 5:
                continue
            feat = df.drop(columns=["capacity"]).values.astype(np.float32)
            soh = (df["capacity"].values / NOM).astype(np.float32)
            rows.append(dict(bid=bid, X=feat, y=soh))
    return rows


def windows(X, y, L):
    xs, ys = [], []
    for i in range(len(y) - L + 1):
        xs.append(X[i:i + L])
        ys.append(y[i + L - 1])
    return np.stack(xs), np.array(ys, dtype=np.float32)


class LSTMNet(nn.Module):
    def __init__(self, nf, hidden=64):
        super().__init__()
        self.lstm = nn.LSTM(nf, hidden, num_layers=1, batch_first=True)
        self.fc = nn.Sequential(nn.Linear(hidden, 32), nn.Tanh(), nn.Linear(32, 1))

    def forward(self, x):
        o, _ = self.lstm(x)
        return self.fc(o[:, -1, :]).squeeze(-1)


def run(seed, data):
    torch.manual_seed(seed); np.random.seed(seed)
    tr, te = [d for d in data if d["bid"] % 5 != 0], [d for d in data if d["bid"] % 5 == 0]

    # 特征 min-max(只用训练集统计量,避免信息泄漏)
    allX = np.concatenate([d["X"] for d in tr], 0)
    lo, hi = allX.min(0), allX.max(0)
    span = np.where(hi - lo < 1e-8, 1.0, hi - lo)

    def prep(ds):
        xs, ys = [], []
        for d in ds:
            X = 2 * (d["X"] - lo) / span - 1
            x, y = windows(X, d["y"], L)
            xs.append(x); ys.append(y)
        return np.concatenate(xs), np.concatenate(ys)

    Xtr, ytr = prep(tr); Xte, yte = prep(te)
    net = LSTMNet(Xtr.shape[2])
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    sch = torch.optim.lr_scheduler.StepLR(opt, step_size=40, gamma=0.5)
    lossf = nn.MSELoss()

    Xtr_t = torch.tensor(Xtr); ytr_t = torch.tensor(ytr)
    n = len(Xtr_t)
    best = 1e9
    for ep in range(EPOCHS):
        net.train()
        perm = torch.randperm(n)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            opt.zero_grad()
            l = lossf(net(Xtr_t[idx]), ytr_t[idx])
            l.backward(); opt.step()
        sch.step()
        if (ep + 1) % 20 == 0:
            net.eval()
            with torch.no_grad():
                pv = net(torch.tensor(Xte)).numpy()
            v = float(np.mean(np.abs(pv - yte)))
            best = min(best, v)
            print(f"    seed{seed} ep{ep+1} testMAE {v*100:.3f}%", flush=True)
    net.eval()
    with torch.no_grad():
        p = net(torch.tensor(Xte)).numpy()
    e = p - yte
    return dict(MAE=float(np.mean(np.abs(e))), MAPE=float(np.mean(np.abs(e / np.abs(yte)))),
                RMSE=float(np.sqrt(np.mean(e ** 2))), R2=float(1 - np.sum(e**2) / np.sum((yte - yte.mean())**2)))


if __name__ == "__main__":
    t0 = time.time()
    data = load_all()
    print(f"载入 {len(data)} 节电池 | 训练 {sum(1 for d in data if d['bid']%5!=0)} / 测试 {sum(1 for d in data if d['bid']%5==0)}", flush=True)
    print("SOH 范围:", round(min(d['y'].min() for d in data), 3), "~", round(max(d['y'].max() for d in data), 3), flush=True)
    res = []
    for s in SEEDS:
        print(f"\n=== seed {s} ===", flush=True)
        r = run(s, data)
        print(f"  MAE {r['MAE']*100:.3f}% | MAPE {r['MAPE']*100:.3f}% | RMSE {r['RMSE']*100:.3f}% | R2 {r['R2']:.3f}", flush=True)
        res.append(r)
    m = {k: float(np.mean([x[k] for x in res])) for k in res[0]}
    sd = {k: float(np.std([x[k] for x in res])) for k in res[0]}
    print("\n" + "=" * 56)
    print(f"LSTM({len(SEEDS)} 个种子)MAPE {m['MAPE']*100:.3f}% ± {sd['MAPE']*100:.3f}% | MAE {m['MAE']*100:.3f}% | RMSE {m['RMSE']*100:.3f}% | R2 {m['R2']:.3f}")
    print(f"对比 PINN 0.656% ± 0.039% | MLP 2.124% ± 0.816%")
    print(f"耗时 {time.time()-t0:.0f}s")
    json.dump(dict(mean=m, std=sd, seeds=SEEDS, runs=res,
                   protocol=dict(L=L, epochs=EPOCHS, batch=BATCH, nom=NOM)),
              open(OUT, "w"), indent=1)

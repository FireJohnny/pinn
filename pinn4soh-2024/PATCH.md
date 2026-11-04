# 复现所需的 2 处改动

原仓库:[wang-fujin/PINN4SOH](https://github.com/wang-fujin/PINN4SOH)(分支 `main`,下载于 2026-10-31)
本仓库**不包含**其源码;下列补丁供读者在自己的副本上应用。

环境差异:原作者用 Python 3.7.10 + PyTorch 1.7.1,本次复现用 **Python 3.11 + PyTorch 2.x**。

---

## 改动 1:`dataloader/dataloader.py` —— pandas 2.x 赋值兼容

**性质**:环境兼容(仅在新版 pandas 上出现)

**现象**

```
TypeError: Invalid value '0 -1.000000 ... Name: cycle index, dtype: float64' for dtype 'int64'
```

**原因**

```python
df.iloc[:, :-1] = f_df     # f_df 是 float,但 df 的列是 int64
```

pandas 1.x 会静默把目标块升格为 float;**pandas 2.x 直接抛 TypeError**。

**补丁**

```diff
-            df.iloc[:,:-1] = f_df
+            # pandas 2.x 不再允许把 float 直接写进 int64 块的 iloc 切片(旧版会静默转换)
+            df = pd.concat([f_df.reset_index(drop=True),
+                            df.iloc[:,-1:].reset_index(drop=True)], axis=1)
```

---

## 改动 2:`main_TJU.py` —— 入口被写成了 `pass`

**性质**:**仓库 bug**,与 Python / PyTorch 版本无关,任何人下载都会遇到

**现象**

```bash
python3 main_TJU.py --epochs 200
# 没有任何输出,进程 exit 0,一个 epoch 都不训练
```

**原因**

```python
if __name__ == '__main__':
    pass          # ← 入口被禁用
```

同目录的 `main_MIT.py` / `main_HUST.py` / `main_XJTU.py` 都是 `main()`,只有 TJU 这个是 `pass`。

**补丁**

```diff
 if __name__ == '__main__':
-    pass
+    main()
```

**影响**:如果不改,论文 Table 1 中 TJU 那一列的结果无法复现 —— 脚本会"成功"地什么都不做。

---

## 复现口径说明

- 训练参数全部用脚本默认值:`epochs=200`、`batch_size=512`、`lr=1e-2`、`alpha=0.5`、`beta=0.01`、网络 3 层×60、正弦激活。
- 测试集划分沿用仓库逻辑:`id % 5 == 0` 的电池进测试集。
- 每次实验取**验证集 MSE 最优轮**的测试指标(与作者 `battery_mean_0` 的口径一致)。
- 仓库**未设置 `torch.manual_seed`**,因此 10 次实验的权重初始化各不相同 —— 这也是本复现的波动大于论文的原因之一(详见 README)。

---

## 两个未解差异(如实记录)

| 项 | 论文 | 本文 | 状态 |
|---|---|---|---|
| MIT 的 RMSE | 0.740% | 0.845%(+14.2%) | **未查明**;候选:torch 版本差异、种子未固定导致的初始化差异 |
| 各数据集波动 | MIT std 0.015% | 0.039%(2.6×) | 同上 |

主指标 MAE / MAPE 均对齐到 7% 以内,这两项不影响主结论。

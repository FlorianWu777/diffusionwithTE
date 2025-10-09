import torch
import matplotlib.pyplot as plt
import numpy as np

# ==== 加载数据 ====
path = "/share/wuhaotian/inference_outputs/test/ensemble_test_c1_old.pt"
tensor = torch.load(path)  # shape: (120, 20, 1, 12, 432, 432)
tensor = tensor.squeeze(2)  # shape: (120, 20, 12, 432, 432)

# ==== 设置阈值 ====
threshold = (0.15 - 0.07842615) / 0.25055954  # ≈ 0.2862

# ==== 准备统计 ====
n_samples, n_ens, n_time, H, W = tensor.shape
counts_ens = np.zeros((n_samples, n_time, n_ens))  # 每个样本，每个时间，每个成员
counts_gt = np.zeros((n_samples, n_time))          # 每个样本，每个时间

# ==== 遍历样本 ====
for i in range(n_samples):
    for t in range(n_time):
        for e in range(n_ens):
            pred = tensor[i, e, t]  # (432, 432)
            counts_ens[i, t, e] = (pred > threshold).sum().item()

        # 真值是第一个 ensemble 成员（你生成时默认这样）
        counts_gt[i, t] = counts_ens[i, t, 0]  # 如果你分离真值了，可从另处加载

def talagrand_histogram(ens: np.ndarray, obs: np.ndarray):
    """
    ens: (N, K)  每个样本的 K 个 ensemble 成员的 scalar 值（如格点数）
    obs: (N,)    每个样本的真实观测值
    return: 相对频率直方图, shape=(K+1,)
    """
    N, K = ens.shape
    ranks = np.zeros(K + 1, dtype=int)

    for i in range(N):
        rank = np.sum(ens[i] < obs[i])  # 有几个成员小于 obs
        ranks[rank] += 1

    return ranks / ranks.sum()


# ==== 示例：对某个时间步绘制 ====
time_idx = 6  # 可调，范围 0 ~ 11
ens_vals = counts_ens[:, time_idx, 1:]  # shape: (N, 19)，忽略第 0 个（真值）
obs_vals = counts_gt[:, time_idx]       # shape: (N,)

# ==== Talagrand 分布 ====
ranks = talagrand_histogram(ens_vals, obs_vals)

# ==== 绘图 ====
plt.figure(figsize=(6, 3))
plt.bar(np.arange(len(ranks))+1, ranks, width=0.8, color="tab:blue")
plt.axhline(1/len(ranks), color="k", lw=1, ls="--")
plt.xlabel("Rank of observation (1 … K)")
plt.ylabel("Relative frequency")
plt.title(f"Talagrand diagram of SIE at 6-Months-Lead-Forecast")
plt.tight_layout()
plt.show()

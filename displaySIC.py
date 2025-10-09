import numpy as np
import matplotlib.pyplot as plt
import os
from zipfile import ZipFile

data = np.load(r"/data/diffusionDemo/conditional_ensemble_members.npy")  # 确保文件在同目录下
output_dir = ".\\geneSIC_6"
os.makedirs(output_dir, exist_ok=True)

for idx in range(500):
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))  # 创建 4x4 网格

    for j in range(16):
        row, col = divmod(j, 4)
        ax = axes[row][col]
        ax.imshow(data[idx, j, 5, :, :], cmap='Blues_r')
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(f"{output_dir}/sample_{idx+1:03d}.png", dpi=100, bbox_inches='tight')
    plt.close()
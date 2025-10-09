import numpy as np
import os
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import TwoSlopeNorm
from tqdm import tqdm

# === 加载数据 ===
data = np.load('./conditional_ensemble_members.npy')  # [N, 50, 6, H, W]
lon = np.load('./lon.npy')  # shape: [W]
lat = np.load('./lat.npy')  # shape: [H]
N_samples, N_members, N_months, H, W = data.shape

# === 创建输出文件夹 ===
output_dir = './ensemble_visualizations'
os.makedirs(output_dir, exist_ok=True)

# === 绘图每个样本的前三个member ===
for i in tqdm(range(N_samples), desc="Plotting 3-member panels"):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), subplot_kw={'projection': ccrs.NorthPolarStereo()})
    
    for m in range(3):
        ax = axes[m]
        sic = data[i, m, 0]  # 第0个月
        ax.set_extent([-180, 180, 60, 90], crs=ccrs.PlateCarree())
        # 颜色标准化
        # norm = TwoSlopeNorm(vmin=0, vcenter=0.5, vmax=1)
        im = ax.pcolormesh(lon, lat, sic, cmap='Blues_r', shading='auto')

        # 添加地理要素
        land = cfeature.NaturalEarthFeature('physical', 'land', '50m',
                                            edgecolor='black', facecolor='lightgray')
        ax.add_feature(land)
        ax.coastlines()

        # 添加子标题
        ax.set_title(f'Member {m+1}')

    # 添加 colorbar（放在底部，跨子图）
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), orientation='horizontal', pad=0.05)
    cbar.set_label('Forecast Value')

    # 保存图像
    filename = f'sample_{i:04d}.png'
    plt.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches='tight')
    plt.close(fig)

print(f"✅ 每个样本的3个成员图已保存在：{output_dir}")

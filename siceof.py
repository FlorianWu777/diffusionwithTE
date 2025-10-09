import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from eofs.standard import Eof
from matplotlib.colors import TwoSlopeNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd
import matplotlib.dates as mdates

# 加载数据：形状为 (T, 432, 432)
# data = np.load(r"D:\IceNet\merged_1979_2021_07.npy")
# data = np.load(r"D:\IceNet\data\network_datasets\dataset1\obs\siconca\abs\sicobs.npy")
# data = np.load(r"D:\IceNet\siconca_pred_r5i1p1f1_r1i1p1f1_r11i1p1f1_r10i1p1f1_r20i1p1f1_r4i1p1f1_r14i1p1f1_r2i1p1f1_r13i1p1f1_r23i1p1f1_r15i1p1f1_r9i1p1f1_r18i1p1f1_r19i1p1f1_r16i1p1f1_r12i1p1f1_r6i1p1f1_r17i1p1f1_r7i1p1f1_r25i1p1f1.npy")
# data = np.load(r"D:\IceNet\20models.npy")[:,:,:,0]/100
# data = np.load(r"D:\IceNet\obs.npy")
# data = np.load(r'.\\sicobs.npy')

lon = np.load('D:\\IceNet\\lon.npy')
lat = np.load('D:\\IceNet\\lat.npy')

data0 = np.load(r"D:\IceNet\ECall.npy")
data = data0.mean(axis=0)
data = np.load(r"C:\Users\Yamasuso\Desktop\ensembleobsforecast.npy")[:,:,:,0]

# data0 = np.load(r"C:\Users\Yamasuso\Desktop\ensembleobsforecast.npy")[:,:,:,0]
data[data>=1]=0.99
pcss = []
eofss = []
for r in range(20):
    data = data0[r,:,:,:]
    T = data.shape[0]

    # Step 1: 找出恒为 0 的点，并将其赋值为 NaN
    mask = np.all(data==0, axis=0)
    data[:, mask] = np.nan  # 将恒为 0 的点替换为 NaN
    data[87,:,:] = (data[87-12,:,:]+data[87+12,:,:])/2
    data[88,:,:] = (data[88-12,:,:]+data[88+12,:,:])/2# Step 2: 去除 NaN 的点，用 eofs 库进行分析时，eofs 会自动处理 NaN
    # 将数据 reshape 为 (T, 空间点)
    reshaped_data = data.reshape(T, -1)

    # Step 3: 去均值处理
    # 按月循环处理，计算每月的气候态并去均值
    anomalies = np.copy(reshaped_data)

    for j in range(12):
        month_indices = np.arange(j, T, 12)
        mean_data = np.nanmean(anomalies[month_indices, :], axis=0)  # 忽略 NaN 进行去均值
        anomalies[month_indices, :] = reshaped_data[month_indices, :] - mean_data

    # Step 4: 使用 eofs 库进行 EOF 分解
    solver = Eof(anomalies)  # 使用 eofs 库处理异常值矩阵
    eofs = solver.eofs(neofs=6)  # 提取前4个 EOF 模式
    pcs = solver.pcs(npcs=6)  # 提取对应的时间主成分
    # pcss.append(pcs)
    # eofss.append(eofs)
    explained_variance = solver.varianceFraction(neigs=6) * 100  # 计算方差贡献率
    error = solver.northTest()
    eigenvalues = solver.eigenvalues()
    # 设置时间序列的时间范围
    dates = pd.date_range(start="1979-01", periods=T, freq="M")

    # Step 5: 绘制前四个模态和时间序列
    fig, axs = plt.subplots(1, 3, figsize=(20, 6), subplot_kw={'projection': ccrs.NorthPolarStereo()})
    fig.subplots_adjust(hspace=0.3, wspace=0.4)
    plt.title('realization '+str(r+1) )

    for i, ax in enumerate(axs.flat[:3]):
        # 获取第 i 个 EOF 模态并重整为 432x432 的空间场
        eof_mode = eofs[i, :].reshape(432, 432)

        # 设置地图范围
        ax.set_extent([-180, 180, 60, 90], crs=ccrs.PlateCarree())

        # 使用 TwoSlopeNorm 设置 colorbar
        norm = TwoSlopeNorm(vmin=-0.01, vcenter=0, vmax=0.01)

        # 使用 pcolormesh 在极地投影上绘制EOF模态，使用 RdBu_r 颜色映射
        im = ax.pcolormesh(lon, lat, eof_mode.T, cmap='RdBu_r', norm=norm, transform=ccrs.PlateCarree())

        # 添加地形和海岸线
        land_feature = cfeature.NaturalEarthFeature(
            'physical', 'land', '50m',
            edgecolor='black', facecolor='lightgray'  # 设置陆地为灰色
        )
        ax.add_feature(land_feature)
        ax.coastlines()

        # Step 6: 绘制对应的时间系数 (主成分时间序列)
        divider = make_axes_locatable(ax)
        time_series_ax = divider.append_axes("top", size="20%", pad=0.5, axes_class=plt.Axes)


        # 绘制对应的时间主成分
        time_series = pcs[:, i]
        time_series_ax.plot(dates, time_series, color='blue')

        # 计算并绘制时间序列的线性趋势
        trend = np.polyfit(mdates.date2num(dates), time_series, 1)
        trend_line = np.polyval(trend, mdates.date2num(dates))
        time_series_ax.plot(dates, trend_line, color='red', linestyle='--')

        # 设置时间刻度
        time_series_ax.xaxis.set_major_locator(mdates.YearLocator(5))
        time_series_ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

        # Step 7: 在模式图下方添加方差贡献率标题
        if eigenvalues[i + 1] < eigenvalues[i] - error[i] or eigenvalues[i + 1] > eigenvalues[i] + error[i]:

            ax.text(0.5, -0.05, f'EOF Mode {i + 1} (Variance: {explained_variance[i]:.2f}%, Significant)',
                    ha='center', va='center', transform=ax.transAxes, fontsize=10)
        else:

            ax.text(0.5, -0.05, f'EOF Mode {i + 1} (Variance: {explained_variance[i]:.2f}%, Not Significant)',
                    ha='center', va='center', transform=ax.transAxes, fontsize=10)

    # Step 8: 添加整体颜色条
    cbar = fig.colorbar(im, ax=axs[:4], orientation='vertical', fraction=0.05, pad=0.04)
    cbar.set_ticks([-0.01, 0, 0.01])
    plt.subplots_adjust(wspace=0.005)

    plt.show()

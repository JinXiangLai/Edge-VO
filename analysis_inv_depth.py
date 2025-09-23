import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os


def read_and_plot_depth_data(filename, result_save_folder: str):
    """
    从文件读取逆深度滤波数据并绘制趋势图
    """
    # 检查文件是否存在
    if not os.path.exists(filename):
        print(f"错误: 文件 {filename} 不存在")
        return

    # 检查必要的列是否存在
    required_columns = [
        'pointId', 'cov', 'invDepth', 'depth', 'trueDepth', 'depthDiff',
        'obvTime'
    ]
    # 读取数据
    try:
        # 跳过注释行，读取CSV格式数据
        df = pd.read_csv(filename,
                         comment='#',
                         names=required_columns,
                         skipinitialspace=True)
        print(f"成功读取数据，共 {len(df)} 行")
        print("数据列名:", df.columns.tolist())
        print("\n数据前5行:")
        print(df.head())
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return

    missing_columns = [
        col for col in required_columns if col not in df.columns
    ]
    if missing_columns:
        print(f"错误: 缺少必要的列: {missing_columns}")
        return

    # 按pointId分组
    grouped = df.groupby('pointId')

    print(f"\n共找到 {len(grouped)} 个不同的pointId")

    # 为每个pointId创建图表
    for point_id, group_data in grouped:
        # 按观测时间排序
        group_data = group_data.sort_values('obvTime')

        # 创建图表
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
        fig.suptitle(f'逆深度滤波分析 - PointID: {point_id}',
                     fontsize=14,
                     fontweight='bold')

        # 子图1: depth和depthDiff的变化趋势，以及真值trueDepth
        obv_times = group_data['obvTime']
        depths = group_data['depth']
        true_depths = group_data['trueDepth']
        depth_diffs = group_data['depthDiff']

        # 绘制深度值
        ax1.plot(obv_times,
                 depths,
                 'b-o',
                 linewidth=2,
                 markersize=4,
                 label='估计深度 (depth)')

        # 绘制真值（如果真值不为0）
        valid_true_depth = true_depths > 1e-6  # 避免接近0的值
        if valid_true_depth.any():
            ax1.plot(obv_times[valid_true_depth],
                     true_depths[valid_true_depth],
                     'g--s',
                     linewidth=2,
                     markersize=6,
                     label='真实深度 (trueDepth)')

        # 绘制深度差异（使用右侧y轴）
        ax1_diff = ax1.twinx()
        ax1_diff.plot(obv_times,
                      depth_diffs,
                      'r-^',
                      linewidth=1,
                      markersize=3,
                      alpha=0.7,
                      label='深度误差 (depthDiff)')

        ax1.set_xlabel('观测次数')
        ax1.set_ylabel('深度值', color='b')
        ax1_diff.set_ylabel('深度误差', color='r')
        ax1.grid(True, alpha=0.3)
        ax1.set_title('深度估计与真值对比')

        # 合并图例
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax1_diff.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')

        # 子图2: 观测次数变化（这里显示累积观测情况）
        cumulative_obs = range(1, len(obv_times) + 1)
        ax2.plot(obv_times,
                 cumulative_obs,
                 'purple-o',
                 linewidth=2,
                 markersize=4)
        ax2.set_xlabel('观测次数')
        ax2.set_ylabel('累积观测帧数')
        ax2.grid(True, alpha=0.3)
        ax2.set_title('观测历史')

        # 子图3: cov的标准差变化趋势
        cov_values = group_data['cov']
        # 计算标准差（cov的平方根）
        std_values = np.sqrt(cov_values)

        ax3.plot(obv_times, std_values, 'orange-s', linewidth=2, markersize=4)
        ax3.set_xlabel('观测次数')
        ax3.set_ylabel('标准差 (sqrt(cov))')
        ax3.grid(True, alpha=0.3)
        ax3.set_title('不确定性变化趋势')

        # 调整布局
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)

        # 保存图片
        output_filename = os.path.join(result_save_folder,
                                       f'depth_analysis_point_{point_id}.png')
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        print(f"已保存图表: {output_filename}")

        # 显示统计信息
        print(f"\nPointID {point_id} 统计:")
        print(f"  观测次数范围: {obv_times.min()} - {obv_times.max()}")
        print(f"  深度范围: {depths.min():.3f} - {depths.max():.3f}")
        print(f"  最终深度: {depths.iloc[-1]:.3f}")
        if valid_true_depth.any():
            print(f"  真实深度: {true_depths[valid_true_depth].iloc[0]:.3f}")
            print(f"  最终误差: {depth_diffs.iloc[-1]:.3f}")
        print(f"  最终不确定性: {std_values.iloc[-1]:.6f}")

        # 显示图片（可选，如果不想显示可以注释掉）
        # plt.show()
        # plt.close()

    # 创建汇总统计
    print("\n" + "=" * 50)
    print("汇总统计:")
    print(f"总数据点数: {len(df)}")
    print(f"唯一PointID数量: {len(grouped)}")

    # 计算每个pointId的平均观测次数
    obs_per_point = df.groupby('pointId')['obvTime'].max()
    print(f"平均每个pointId的观测次数: {obs_per_point.mean():.2f}")

    # 计算深度误差统计（只考虑有真值的数据）
    valid_data = df[df['trueDepth'] > 1e-6]
    if len(valid_data) > 0:
        print(f"有真值的数据点: {len(valid_data)}")
        print(f"平均深度误差: {valid_data['depthDiff'].abs().mean():.3f}")
        print(f"最大深度误差: {valid_data['depthDiff'].abs().max():.3f}")
        print(f"深度误差标准差: {valid_data['depthDiff'].std():.3f}")


def main(inv_depth_file_path: str, result_save_folder: str):

    # 如果文件不存在，创建一个示例文件（用于测试）
    if not os.path.exists(inv_depth_file_path):
        print(f"Error file: {inv_depth_file_path} no exists!")
        return

    # 读取数据并绘制图表
    read_and_plot_depth_data(inv_depth_file_path, result_save_folder)


if __name__ == "__main__":
    inv_depth_file_path = "/home/ht/Opencv_Ceres_Eigen_example/Edge-VO/build/test.csv"
    result_save_folder = "/home/ht/Opencv_Ceres_Eigen_example/Edge-VO/build"
    if not os.path.exists(result_save_folder):
        os.makedirs(result_save_folder)

    main(inv_depth_file_path, result_save_folder)

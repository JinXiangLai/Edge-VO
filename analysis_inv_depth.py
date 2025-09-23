import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# 设置matplotlib使用英文字体，避免中文问题
plt.rcParams['font.family'] = 'DejaVu Sans'  # 使用系统英文字体
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

def read_and_plot_depth_data(filename, save_folder: str):
    """
    Read inverse depth filter data and plot trends
    """
    # 检查文件是否存在
    if not os.path.exists(filename):
        print(f"Error: File {filename} does not exist")
        return
    
    # 创建保存文件夹
    if save_folder is None:
        save_folder = "depth_analysis_results"
    os.makedirs(save_folder, exist_ok=True)
    
    # 读取数据
    try:
        # 如果上面的方式失败，尝试手动处理
        with open(filename, 'r') as f:
            lines = [line.strip() for line in f if not line.startswith('#') and line.strip()]
        
        data = []
        for line in lines:
            parts = line.split(',')
            if len(parts) >= 7:
                data.append([part.strip() for part in parts])
        
        df = pd.DataFrame(data, columns=['pointId', 'cov', 'invDepth', 'depth', 'trueDepth', 'depthDiff', 'obvTime'])
        
        # 转换数据类型
        df = df.apply(pd.to_numeric, errors='coerce')

        print(f"Successfully read data, total {len(df)} rows")
        print("Column names:", df.columns.tolist())
        print("\nFirst 5 rows:")
        print(df.head())
        
    except Exception as e:
        print(f"Error reading file: {e}")
        return
    
    # 检查必要的列是否存在
    required_columns = ['pointId', 'cov', 'invDepth', 'depth', 'trueDepth', 'depthDiff', 'obvTime']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"Error: Missing required columns: {missing_columns}")
        print(f"Actual columns: {df.columns.tolist()}")
        return
    
    # 按pointId分组
    grouped = df.groupby('pointId')
    
    print(f"\nFound {len(grouped)} different pointIds")
    
    # 为每个pointId创建图表
    for point_id, group_data in grouped:
        # 按观测时间排序
        group_data = group_data.sort_values('obvTime')
        
        # 创建图表
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
        fig.suptitle(f'Inverse Depth Analysis - PointID: {point_id}', fontsize=14, fontweight='bold')
        
        # 子图1: depth和depthDiff的变化趋势，以及真值trueDepth
        obv_times = group_data['obvTime']
        depths = group_data['depth']
        true_depths = group_data['trueDepth']
        depth_diffs = group_data['depthDiff']
        print("point_id: ", point_id)
        print("depths: ", depths.values)
        print("depth_diffs: ", depth_diffs.values)
        print("true_depths: ", true_depths.values)
        
        # 绘制深度值
        ax1.plot(obv_times, depths, 'b-o', linewidth=2, markersize=4, label='Estimated Depth')
        
        # 绘制真值（如果真值不为0）
        valid_true_depth = true_depths > 1e-6  # 避免接近0的值，这是找下标吧？
        if valid_true_depth.any():
            ax1.plot(obv_times[valid_true_depth], true_depths[valid_true_depth], 
                    'g--', marker='s', linewidth=2, markersize=6, label='True Depth')

            # 在图表合适位置添加真值标签（避免遮挡）
            first_true_time = obv_times.iloc[0]
            first_true_value = true_depths.iloc[0]
            y_range = ax1.get_ylim()
            label_y = first_true_value + (y_range[1] - y_range[0]) * 0.02  # 稍微向上偏移
            
            ax1.text(first_true_time, label_y, f'True Depth: {first_true_value:.3f}', 
                    ha='right', va='bottom', fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8))

        
        # 绘制深度差异（使用右侧y轴）
        ax1_diff = ax1.twinx()
        ax1_diff.plot(obv_times, depth_diffs, 'r-', marker='^', linewidth=1, markersize=3, 
                     alpha=0.7, label='Depth Error')
        
        ax1.set_xlabel('Observation Times')
        ax1.set_ylabel('Depth Value', color='b')
        ax1_diff.set_ylabel('Depth Error', color='r')
        ax1.grid(True, alpha=0.3)
        ax1.set_title('Depth Estimation vs Ground Truth')
        
        # 合并图例
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax1_diff.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')
        
        # 子图2: 观测次数变化
        cumulative_obs = range(1, len(obv_times) + 1)
        ax2.plot(obv_times, cumulative_obs, 'purple', marker='o', linewidth=2, markersize=4)
        ax2.set_xlabel('Observation Times')
        ax2.set_ylabel('Cumulative Frames')
        ax2.grid(True, alpha=0.3)
        ax2.set_title('Observation History')
        
        # 子图3: cov的标准差变化趋势
        cov_values = group_data['cov']
        # 计算标准差（cov的平方根）
        std_values = np.sqrt(cov_values)
        
        ax3.plot(obv_times, std_values, 'orange', marker='s', linewidth=2, markersize=4)
        ax3.set_xlabel('Observation Times')
        ax3.set_ylabel('Standard Deviation')
        ax3.grid(True, alpha=0.3)
        ax3.set_title('Uncertainty Trend')
        
        # 调整布局
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)
        
        # 保存图片
        output_filename = os.path.join(save_folder, f'depth_analysis_point_{point_id}.png')
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_filename}")
        
        # 显示统计信息
        print(f"\nPointID {point_id} Statistics:")
        print(f"  Observation range: {obv_times.min()} - {obv_times.max()}")
        print(f"  Depth range: {depths.min():.3f} - {depths.max():.3f}")
        print(f"  Final depth: {depths.iloc[-1]:.3f}")
        if valid_true_depth.any():
            true_depth_val = true_depths[valid_true_depth].iloc[0] if valid_true_depth.any() else 0
            print(f"  True depth: {true_depth_val:.3f}")
            print(f"  Final error: {depth_diffs.iloc[-1]:.3f}")
        print(f"  Final uncertainty: {std_values.iloc[-1]:.6f}")
        
        plt.close()
    
    # 创建汇总统计图
    create_summary_plot(df, save_folder)
    
    # 创建汇总统计
    print("\n" + "="*50)
    print("Summary Statistics:")
    print(f"Total data points: {len(df)}")
    print(f"Unique PointIDs: {len(grouped)}")
    
    # 计算深度误差统计（只考虑有真值的数据）
    valid_data = df[df['trueDepth'] > 1e-6]
    if len(valid_data) > 0:
        print(f"Data points with ground truth: {len(valid_data)}")
        print(f"Mean absolute error: {valid_data['depthDiff'].abs().mean():.3f}")
        print(f"Max absolute error: {valid_data['depthDiff'].abs().max():.3f}")
        print(f"Error standard deviation: {valid_data['depthDiff'].std():.3f}")

def create_summary_plot(df, save_folder):
    """
    Create summary plots for all data
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Inverse Depth Filter Summary Analysis', fontsize=16, fontweight='bold')
    
    # 子图1: 深度误差分布
    valid_data = df[df['trueDepth'] > 1e-6]
    if len(valid_data) > 0:
        ax1.hist(valid_data['depthDiff'].abs(), bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_xlabel('Absolute Depth Error')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Depth Error Distribution')
        ax1.grid(True, alpha=0.3)
    
    # 子图2: 不确定性分布
    std_values = np.sqrt(df['cov'])
    ax2.hist(std_values, bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
    ax2.set_xlabel('Standard Deviation')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Uncertainty Distribution')
    ax2.grid(True, alpha=0.3)
    
    # 子图3: 观测次数分布
    ax3.hist(df['obvTime'], bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
    ax3.set_xlabel('Observation Times')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Observation Times Distribution')
    ax3.grid(True, alpha=0.3)
    
    # 子图4: 深度值分布
    ax4.hist(df['depth'], bins=20, alpha=0.7, color='gold', edgecolor='black')
    ax4.set_xlabel('Depth Value')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Depth Value Distribution')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    
    output_filename = os.path.join(save_folder, 'depth_analysis_summary.png')
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Saved summary plot: {output_filename}")
    plt.close()

def main(inv_depth_file_path: str, save_folder: str):

    # 如果文件不存在，创建一个示例文件（用于测试）
    if not os.path.exists(inv_depth_file_path):
        print(f"Error file: {inv_depth_file_path} no exists!")
        return

    # 读取数据并绘制图表
    read_and_plot_depth_data(inv_depth_file_path, save_folder)


if __name__ == "__main__":
    inv_depth_file_path = "/home/laijinxiang/edge-slam/Edge-VO/build/kf_1_depth_uncertainty.csv"
    save_folder = "/home/laijinxiang/edge-slam/Edge-VO/build/result"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    main(inv_depth_file_path, save_folder)

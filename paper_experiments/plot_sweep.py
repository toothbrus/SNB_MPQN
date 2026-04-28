import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ================= 配置 =================
CSV_FILE = "execution_time_pc_sweep.csv"  # 请确保文件名正确
Y_COLUMN = "exec_time"
# ERROR_COLUMN = "exec_time_std"
X_COLUMN = "p_c"
OUTPUT_PREFIX = "pc_sweep_exec_time"
# =========================================

df = pd.read_csv(CSV_FILE)

# 标签：直接使用 network 的大写形式（包括 MPQN, QMSN, STATIC_LINE, STATIC_GRID）
df['label'] = df['network'].str.upper()

# 为了让图例更易读，可以自定义顺序（可选）
label_order = ['MPQN', 'QMSN', 'STATIC_LINE', 'STATIC_GRID']

# 为常见网络指定颜色，使图表更清晰
colors = {
    'MPQN': '#1f77b4',        # 蓝色
    'QMSN': '#ff7f0e',        # 橙色
    'STATIC_LINE': '#2ca02c', # 绿色
    'STATIC_GRID': '#d62728'  # 红色
}

circuits = df['circuit'].unique()
plt.style.use('seaborn-v0_8-whitegrid')

for circuit in circuits:
    circuit_df = df[df['circuit'] == circuit].copy()
    
    # 按指定顺序排序（存在的标签）
    existing_labels = [l for l in label_order if l in circuit_df['label'].unique()]
    
    plt.figure(figsize=(8, 5))
    
    for label in existing_labels:
        group = circuit_df[circuit_df['label'] == label].sort_values(by=X_COLUMN)
        if group.empty:
            continue
        plt.errorbar(
            group[X_COLUMN], group[Y_COLUMN],
            # yerr=group[ERROR_COLUMN],
            marker='o', markersize=6, capsize=5,
            label=label, linewidth=2,
            color=colors.get(label, None)  # 如果未定义，自动使用默认颜色
        )
    
    plt.xlabel('Photon Collection Rate $p_c$')
    plt.ylabel('Execution Time (μs)')
    plt.yscale('log')      # 关键：对数刻度，避免静态网络数据掩盖其他曲线
    plt.legend()
    plt.tight_layout()
    
    filename = f"{OUTPUT_PREFIX}_{circuit}.pdf"
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Saved plot for {circuit} -> {filename}")

print("All plots generated.")
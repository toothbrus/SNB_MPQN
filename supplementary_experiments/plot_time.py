import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ================= 可配置参数 =================
CSV_FILE = "experiment_results_raw.csv"   # 你的 summary CSV
OUTPUT_PREFIX = "time_bar"                    # 输出图片前缀
# =============================================

df = pd.read_csv(CSV_FILE)

# 创建显示标签：区分 QMSN 的两种模式
def make_label(row):
    if row['network'] == 'qmsn':
        mode = row.get('mode', 'N/A')
        if mode == 'reconfig':
            return 'QMSN (reconfig)'
        elif mode == 'timeslot':
            return 'QMSN (timeslot)'
        else:
            return 'QMSN'
    else:
        return row['network'].upper()

df['label'] = df.apply(make_label, axis=1)

# 期望的标签顺序（所有网络）
label_order = ['MPQN', 'QMSN (timeslot)', 'QMSN (reconfig)', 'STATIC_LINE', 'STATIC_GRID']
# 只保留实际存在的标签
existing_labels = [l for l in label_order if l in df['label'].unique()]

circuits = df['circuit'].unique()

plt.style.use('seaborn-v0_8-whitegrid')

for circuit in circuits:
    circuit_df = df[df['circuit'] == circuit]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(existing_labels))
    width = 0.6
    colors = plt.cm.tab10.colors  # 使用更多颜色区分

    values = []
    errors = []
    for label in existing_labels:
        match = circuit_df[circuit_df['label'] == label]
        if not match.empty:
            values.append(match['total_time_mean'].values[0])
            errors.append(match['total_time_std'].values[0])
        else:
            values.append(0)
            errors.append(0)

    bars = ax.bar(x, values, width, yerr=errors, capsize=6,
                  color=[colors[i % len(colors)] for i in range(len(existing_labels))])

    # 标注数值
    for bar, val in zip(bars, values):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02 * max(values),
                    f'{val:.1f}', ha='center', va='bottom', fontsize=8, rotation=0)

    ax.set_xticks(x)
    ax.set_xticklabels(existing_labels, rotation=30, ha='right')
    ax.set_ylabel('Total Execution Time (μs)')
    ax.set_title(f'Execution Time Comparison for {circuit}')
    plt.tight_layout()
    filename = f"{OUTPUT_PREFIX}_{circuit}_1.pdf"
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Saved time plot for {circuit} -> {filename}")

print("All time bar plots generated.")
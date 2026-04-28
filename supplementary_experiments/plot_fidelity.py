import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ================= 可配置参数 =================
CSV_FILE = "debug_tinit_raw.csv"   # 你的 summary CSV
OUTPUT_PREFIX = "fidelity_relative_bar"       # 输出图片前缀
# =============================================

df = pd.read_csv(CSV_FILE)

# 创建显示标签
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

# 期望标签顺序（所有网络）
label_order = ['MPQN', 'QMSN (timeslot)', 'QMSN (reconfig)', 'STATIC_LINE', 'STATIC_GRID']
existing_labels = [l for l in label_order if l in df['label'].unique()]

circuits = df['circuit'].unique()

plt.style.use('seaborn-v0_8-whitegrid')

for circuit in circuits:
    circuit_df = df[df['circuit'] == circuit].copy()
    
    # 计算真实保真度值
    fid_values = []
    fid_errors = []
    actual_labels = []
    for label in existing_labels:
        match = circuit_df[circuit_df['label'] == label]
        if not match.empty:
            val = match['computational_fidelity_mean'].values[0]
            err = match['computational_fidelity_std'].values[0]
            fid_values.append(val)
            fid_errors.append(err)
            actual_labels.append(label)
        else:
            fid_values.append(0.0)
            fid_errors.append(0.0)
    
    # 找到最大保真度（基准值）
    max_fid = max(fid_values) if fid_values else 1.0
    if max_fid <= 0:   # 避免除零
        max_fid = 1.0

    # 计算相对值
    relative_values = [v / max_fid for v in fid_values]
    # 相对误差（近似）
    relative_errors = [e / max_fid for e in fid_errors]

    # 绘制柱子
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(actual_labels))
    width = 0.6
    colors = plt.cm.tab10.colors

    bars = ax.bar(x, relative_values, width, yerr=relative_errors, capsize=6,
                  color=[colors[i % len(colors)] for i in range(len(actual_labels))])

    # 标注原始保真度数值（科学计数法）和相对值（可选）
    for bar, orig_val, rel_val in zip(bars, fid_values, relative_values):
        height = bar.get_height()
        # 在柱子上方显示原始保真度（科学计数法）
        annotation = f'{orig_val:.2e}'
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                annotation, ha='center', va='bottom', fontsize=7, rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels(actual_labels, rotation=30, ha='right')
    ax.set_ylabel('Relative Fidelity (normalized to max)')
    ax.set_title(f'Fidelity Comparison for {circuit}\n(Max fidelity = {max_fid:.2e})')
    # 添加基准线
    ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=0.8)
    ax.set_ylim(0, 1.15)  # 留出标注空间
    plt.tight_layout()
    filename = f"{OUTPUT_PREFIX}_{circuit}.pdf"
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Saved fidelity plot for {circuit} -> {filename}")

print("All fidelity relative bar plots generated.")
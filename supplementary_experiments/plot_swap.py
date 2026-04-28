import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys

# ================= 可配置参数 =================
CSV_FILE = "myresults_pent_raw.csv"   # 你的 summary CSV 文件
Y_COLUMN = "total_time_mean"            # 或 "computational_fidelity_grand_mean"
ERROR_COLUMN = "total_time_std"         # 或 "computational_fidelity_grand_std"
PARAM_NAME = "p_ent"                         # 扫描的参数名（例如 "t_deco", "t_init", "p_ent"）
OUTPUT_PREFIX = f"{PARAM_NAME}_sweep_plot"                  # 输出图片的前缀
# =============================================

# 读取数据
df = pd.read_csv(CSV_FILE)
# # 隐藏静态网络
# df = df[~df['network'].isin(['static_line', 'static_grid'])]

# 检查必要列是否存在
if PARAM_NAME not in df.columns:
    raise ValueError(f"参数列 '{PARAM_NAME}' 不存在于 CSV 中，可用列名：{list(df.columns)}")

# 创建统一标签列（区分网络和模式）
def make_label(row):
    if row['network'] == 'qmsn':
        mode = row.get('mode', 'N/A')
        if mode == 'timeslot':
            return 'QMSN (timeslot)'
        elif mode == 'reconfig':
            return 'QMSN (reconfig)'
        else:
            return 'QMSN'
    else:
        return row['network'].upper()
df['label'] = df.apply(make_label, axis=1)

# 获取所有电路名称
circuits = df['circuit'].unique()

# 设置 seaborn 风格（可选）
sns.set_theme(style="whitegrid")

for circuit in circuits:
    # 筛选当前电路的数据
    circuit_df = df[df['circuit'] == circuit].copy()

    # 创建图形
    plt.figure(figsize=(10, 6))

    # 按 label 分组，为每种网络模式画折线（带误差棒）
    for label, group in circuit_df.groupby('label'):
        # 按参数值排序，确保线条正确连接
        group = group.sort_values(by=PARAM_NAME)
        plt.errorbar(
            group[PARAM_NAME], group[Y_COLUMN],
            yerr=group[ERROR_COLUMN],
            marker='o', capsize=5, label=label
        )

    # 标签和标题
    plt.xlabel(PARAM_NAME)
    plt.ylabel("Execution Time (μs)")
    # plt.title(f'Effect of {PARAM_NAME} on {Y_COLUMN} for circuit: {circuit}')
    plt.legend()
    plt.tight_layout()

    # 保存每个电路单独的图片
    filename = f"{OUTPUT_PREFIX}_{circuit}.pdf"
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Saved plot for {circuit} -> {filename}")

print("All plots generated.")
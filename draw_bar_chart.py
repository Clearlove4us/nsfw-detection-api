import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. 字体与显示配置 (解决中文方块问题)
# ==========================================
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

# ==========================================
# 2. 从你的两张截图中提取的真实数据
# ==========================================
# 指标名称
labels = ['正常图片\n精确率(Precision)', '正常图片\n召回率(Recall)',
          '违规图片\n精确率(Precision)', '违规图片\n召回率(Recall)',
          '总体准确率\n(Accuracy)']

# 图1数据 (优化前)
v3_scores = [0.8937, 0.9418, 0.9416, 0.8934, 0.9170]
# 图2数据 (优化后/终极版)
v4_scores = [0.9423, 0.9698, 0.9704, 0.9435, 0.9563]

x = np.arange(len(labels))  # 标签在 X 轴的物理位置
width = 0.35  # 柱子的宽度

# ==========================================
# 3. 开始绘制高质量柱状图
# ==========================================
fig, ax = plt.subplots(figsize=(12, 6))

# 绘制两组柱状图，采用高级学术配色 (莫兰迪蓝 vs 活力橙)
rects1 = ax.bar(x - width/2, v3_scores, width, label='Baseline', color='#5C81D6', alpha=0.9)
rects2 = ax.bar(x + width/2, v4_scores, width, label='LoRA微调', color='#E67E22', alpha=0.9)

# 添加必要的文本、标题和轴标签
ax.set_ylabel('评估分数 (Score)', fontsize=12, fontweight='bold')
ax.set_title('基线模型与LoRA微调模型在动漫测试集上的性能对比', fontsize=16, pad=20, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=11)
ax.legend(loc='upper left', fontsize=11)

# ⚠️ 核心技巧：限制 Y 轴的显示范围 (比如0.92 到 0.98)
# 如果从 0 开始画，两组柱子看起来会一样高。截取范围能清晰展示提升幅度。
ax.set_ylim(0, 1.15)

# 添加横向虚线网格，方便肉眼对齐
ax.grid(axis='y', linestyle='--', alpha=0.6)

# ==========================================
# 4. 定义自动添加顶部数值标签的函数
# ==========================================
def autolabel(rects):
    """在每个柱子上方附加一个文本标签，显示其高度 (具体数值)"""
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.4f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 5),  # 垂直向上偏移 5 个像素
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=11, fontweight='bold')

# 调用函数为两组柱子加上数值
autolabel(rects1)
autolabel(rects2)

# ==========================================
# 5. 布局自适应与高清保存
# ==========================================
fig.tight_layout()
save_path = "./metrics_comparison_bar_chart.png"
plt.savefig(save_path, dpi=300)  # 300dpi 满足绝大多数高校毕业论文的高清打印要求
print(f"📊 完美柱状图已生成并保存至: {save_path}")

#plt.show()  # 运行后可以直接在屏幕上预览

import json
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "WenQuanYi Micro Hei"]
plt.rcParams["axes.unicode_minus"] = False

# 1. 读取你训练时保存的日志文件
with open("./falconsai_lora_anime_v4/training_logs_v4.json", "r") as f:
    log_history = json.load(f)

# 2. 解析数据 (和你的原代码一样)
train_epochs, train_loss = [], []
eval_epochs, eval_loss, eval_accuracy = [], [], []

for log in log_history:
    if 'loss' in log and 'epoch' in log:
        train_epochs.append(log['epoch'])
        train_loss.append(log['loss'])
    elif 'eval_loss' in log and 'epoch' in log:
        eval_epochs.append(log['epoch'])
        eval_loss.append(log['eval_loss'])
        eval_accuracy.append(log['eval_accuracy'])

# 3. 开始画图 (这里可以随意调整各种论文所需的参数)
plt.figure(figsize=(12, 5))
# ... 下面的画图代码和你的原代码完全一致 ...
# 画图：双轴子图
plt.figure(figsize=(12, 5))

# 子图1：Loss 曲线
plt.subplot(1, 2, 1)
plt.plot(train_epochs, train_loss, label='训练损失', color='blue', alpha=0.6)
plt.plot(eval_epochs, eval_loss, label='验证损失', color='red', marker='o')
plt.title('训练与验证损失变化曲线')
plt.xlabel('训练轮次(Epochs)')
plt.ylabel('损失值(Loss)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

# 子图2：Accuracy 曲线
plt.subplot(1, 2, 2)
plt.plot(eval_epochs, eval_accuracy, label='验证准确率', color='green', marker='s')
plt.title('验证集准确率变化曲线')
plt.xlabel('训练轮次(Epochs)')
plt.ylabel('准确率（Accuracy）')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plot_path = "./falconsai_lora_anime_v4/training_curves_v4.png"
plt.savefig(plot_path, dpi=300)  # 高清保存
print(f"📊 训练曲线已保存至: {plot_path}")
# ==========================================

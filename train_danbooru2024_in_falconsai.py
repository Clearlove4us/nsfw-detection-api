import os
import torch
import numpy as np
import evaluate
import json
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import (
    ViTImageProcessor,
    ViTForImageClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback  # 新增：引入早停回调模块
)
import torchvision.transforms as transforms
from peft import LoraConfig, get_peft_model
from huggingface_hub import login

# ==========================================
# 0. 全局变量与函数定义
# ==========================================
model_name = "FalconsAI/nsfw_image_detection"

processor = ViTImageProcessor.from_pretrained(model_name)

train_augmentation = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.1, contrast=0.1)
])


def train_transform(example_batch):
    augmented_images = [train_augmentation(x.convert("RGB")) for x in example_batch['image']]
    inputs = processor(augmented_images, return_tensors='pt')
    inputs['labels'] = example_batch['label']
    return inputs


def eval_transform(example_batch):
    images = [x.convert("RGB") for x in example_batch['image']]
    inputs = processor(images, return_tensors='pt')
    inputs['labels'] = example_batch['label']
    return inputs


def compute_metrics(eval_pred):
    metric = evaluate.load("accuracy")
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return metric.compute(predictions=predictions, references=eval_pred.label_ids)


# ==========================================
# ⚠️ 核心执行入口
# ==========================================
if __name__ == '__main__':
    # 🌟 1. GPU 硬件检测与打印
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 40)
    if device == "cuda":
        print(f"🚀 硬件检测成功！正在使用 GPU 进行加速。")
        print(f"🎮 显卡型号: {torch.cuda.get_device_name(0)}")
        print(f"💾 显卡总显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("⚠️ 警告：未检测到可用的 GPU，目前正在使用纯 CPU 龟速炼丹！请检查虚拟环境。")
    print("=" * 40)

    # 2. Hugging Face 身份认证
    login(token="")

    # 3. 路径配置
    data_dir = r"D:\python-learning\FalconsAI_NSFW\danbooru_dataset"

    # 4. 加载数据集
    print("🚀 正在加载数据集...")
    dataset = load_dataset("imagefolder", data_files={
        "train": f"{data_dir}/train/**",
        "validation": f"{data_dir}/val/**",
        "test": f"{data_dir}/test/**"
    })

    dataset["train"].set_transform(train_transform)
    dataset["validation"].set_transform(eval_transform)
    dataset["test"].set_transform(eval_transform)

    # 5. 加载预训练模型
    print("🧠 正在加载原版 FalconsAI 模型...")
    model = ViTForImageClassification.from_pretrained(model_name)

    # 6. 注入 LoRA
    print("✨ 正在注入 LoRA 模块进行二次元领域自适应...")
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["query", "value"],
        lora_dropout=0.01,
        bias="none",
        modules_to_save=["classifier"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 7. 训练超参数
    training_args = TrainingArguments(
        output_dir="./falconsai_lora_anime_v3",
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        gradient_accumulation_steps=1,
        fp16=True,
        eval_strategy="epoch",
        save_strategy="epoch",

        # 👇 核心防过拟合修改区 👇
        learning_rate=3e-5,  # 1. 降低学习率（原来是5e-4），让模型学得更细腻
        weight_decay=1e-4,  # 2. 新增 L2 正则化，惩罚过大的权重，防止死记硬背
        lr_scheduler_type="cosine",  # 3. 新增余弦退火学习率调度，让学习率平滑下降

        # 👆 核心防过拟合修改区 👆

        num_train_epochs=10,
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="eval_accuracy",
        greater_is_better=True,
        remove_unused_columns=False,
        dataloader_num_workers=2,
    )

    # 8. 启动训练器 (挂载早停回调)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=processor,
        compute_metrics=compute_metrics,
        #callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    print("🔥 开始炼丹！显卡风扇起飞预警...")
    trainer.train()

    # ==========================================
    # 📈 9. 提取日志并绘制训练曲线 (用于论文第三章)
    # ==========================================
    print("📈 正在提取训练日志并绘制收敛曲线...")
    log_history = trainer.state.log_history

    # 将原始日志保存为 JSON
    with open("./falconsai_lora_anime_v3/training_logs_v3.json", "w") as f:
        json.dump(log_history, f, indent=4)

    train_epochs, train_loss = [], []
    eval_epochs, eval_loss, eval_accuracy = [], [], []

    # 解析日志
    for log in log_history:
        if 'loss' in log and 'epoch' in log:
            train_epochs.append(log['epoch'])
            train_loss.append(log['loss'])
        elif 'eval_loss' in log and 'epoch' in log:
            eval_epochs.append(log['epoch'])
            eval_loss.append(log['eval_loss'])
            eval_accuracy.append(log['eval_accuracy'])

    # 画图：双轴子图
    plt.figure(figsize=(12, 5))

    # 子图1：Loss 曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_epochs, train_loss, label='Training Loss', color='blue', alpha=0.6)
    plt.plot(eval_epochs, eval_loss, label='Validation Loss', color='red', marker='o')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    # 子图2：Accuracy 曲线
    plt.subplot(1, 2, 2)
    plt.plot(eval_epochs, eval_accuracy, label='Validation Accuracy', color='green', marker='s')
    plt.title('Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plot_path = "./falconsai_lora_anime_v3/training_curves_v3.png"
    plt.savefig(plot_path, dpi=300)  # 高清保存
    print(f"📊 训练曲线已保存至: {plot_path}")
    # ==========================================

    # 10. 最终盲测与保存
    print("🧪 正在使用 Test 测试集进行最终打分...")
    test_results = trainer.evaluate(dataset["test"])
    print(f"🎉 FalconsAI(LoRA微调后) 最终测试集准确率: {test_results['eval_accuracy']:.4f}")

    print("\n🔍 正在生成深度体检报告 (Classification Report)...")
    predictions = trainer.predict(dataset["test"])
    y_pred = np.argmax(predictions.predictions, axis=1)
    y_true = predictions.label_ids

    from sklearn.metrics import classification_report, confusion_matrix

    report = classification_report(
        y_true,
        y_pred,
        target_names=["normal", "porn"],
        digits=4
    )
    print("\n" + "=" * 30)
    print("模型结业报告 (Test Set)")
    print("=" * 30)
    print(report)
    print("=" * 30)

    print("\n📊 混淆矩阵 (Confusion Matrix):")
    print(confusion_matrix(y_true, y_pred))

    model.save_pretrained("./final_falconsai_lora_v3")
    processor.save_pretrained("./final_falconsai_lora_v3")
    print("💾 V2版本权重已永久保存至 ./final_falconsai_lora_v3")

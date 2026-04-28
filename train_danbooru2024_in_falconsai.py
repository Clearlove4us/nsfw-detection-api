import os
import torch
import numpy as np
import evaluate
from datasets import load_dataset
from transformers import (
    ViTImageProcessor,
    ViTForImageClassification,
    TrainingArguments,
    Trainer
)
import torchvision.transforms as transforms
from peft import LoraConfig, get_peft_model
from huggingface_hub import login  # 引入 HF 登录模块

# ==========================================
# 0. 全局变量与函数定义
# (必须放在 if 外面，保证 DataLoader 子进程能找到并加载它们)
# ==========================================
model_name = "FalconsAI/nsfw_image_detection"

# Processor 和 Augmentation 是轻量级的，放在全局初始化供所有子进程使用
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
# ⚠️ 核心执行入口：Windows 多进程保护罩
# ==========================================
if __name__ == '__main__':
    # 0. Hugging Face 身份认证 (消除警告，满速下载)
    # 请务必将下面这串 hf_ 开头的字符串替换为你自己的 Token！
    login(token="私钥")

    # 1. 路径配置
    data_dir = r"D:\python-learning\FalconsAI_NSFW\danbooru_dataset"

    # 2. 加载数据集
    print("🚀 正在加载数据集...")
    dataset = load_dataset("imagefolder", data_files={
        "train": f"{data_dir}/train/**",
        "validation": f"{data_dir}/val/**",
        "test": f"{data_dir}/test/**"
    })

    # 将放在全局的预处理函数挂载到数据集上
    dataset["train"].set_transform(train_transform)
    dataset["validation"].set_transform(eval_transform)
    dataset["test"].set_transform(eval_transform)

    # 3. 加载预训练模型
    print("🧠 正在加载原版 FalconsAI 模型...")
    model = ViTForImageClassification.from_pretrained(model_name)

    # 4. 注入 LoRA
    print("✨ 正在注入 LoRA 模块进行二次元领域自适应...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=["query", "value"],
        lora_dropout=0.1,
        bias="none",
        modules_to_save=["classifier"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 5. 训练超参数
    training_args = TrainingArguments(
        output_dir="./falconsai_lora_anime",
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        gradient_accumulation_steps=1,
        fp16=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-4,
        num_train_epochs=5,
        logging_steps=50,
        load_best_model_at_end=True,
        remove_unused_columns=False,
        dataloader_num_workers=2,
    )

    # 6. 启动训练器
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=processor,
        compute_metrics=compute_metrics,
    )

    print("🔥 开始炼丹！显卡风扇起飞预警...")
    trainer.train()

    # 7. 最终盲测与保存
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
    print("\n" + "="*30)
    print("模型结业报告 (Test Set)")
    print("="*30)
    print(report)
    print("="*30)

    print("\n📊 混淆矩阵 (Confusion Matrix):")
    print(confusion_matrix(y_true, y_pred))

    model.save_pretrained("./final_falconsai_lora")
    processor.save_pretrained("./final_falconsai_lora")
    print("💾 专门针对二次元优化的 LoRA 权重已永久保存至 ./final_falconsai_lora")
    #调参对比选换文件夹路径，二次进修选加载已有权重。
    # # 5. 训练超参数
    # training_args = TrainingArguments(
    #     output_dir="./falconsai_lora_anime_v2",  # 👈 加上 _v2
    #     # ... (你修改的新参数，比如 learning_rate=1e-3 等)
    #     # ...
    # )
    #
    # # ... (中间代码不变) ...
    #
    # # 7. 最终盲测与保存
    # model.save_pretrained("./final_falconsai_lora_v2")  # 👈 加上 _v2
    # processor.save_pretrained("./final_falconsai_lora_v2")  # 👈 加上 _v2
    # print("💾 V2版本权重已保存至 ./final_falconsai_lora_v2")
    #
    # # 替换掉原来的注入空白 LoRA 的代码：
    # # model = get_peft_model(model, lora_config)
    #
    # # 改为直接加载你已经保存的成品：
    # print("✨ 正在加载上一次的极品权重继续深造...")
    # model = PeftModel.from_pretrained(base_model, "./final_falconsai_lora", is_trainable=True)

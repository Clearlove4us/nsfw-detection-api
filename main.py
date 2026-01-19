
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from transformers import pipeline
from PIL import Image
import io
import torch
import logging

# 设置日志，方便查看运行状态
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 1. 创建FastAPI应用实例
app = FastAPI(title="NSFW内容检测API", description="基于FalconsAI模型的NSFW内容检测系统")

# 2. 在启动时加载模型到GPU
device = 0 if torch.cuda.is_available() else -1
logger.info(f"正在使用设备: {'GPU' if device == 0 else 'CPU'}")
try:
    # 使用pipeline加载模型，这是Hugging Face最简化的方式
    classifier = pipeline("image-classification",
                          model="Falconsai/nsfw_image_detection",
                          device=device)
    # classifier("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hub/parrots.png")
    logger.info("✅ FalconsAI 模型加载成功！")
except Exception as e:
    logger.error(f"❌ 模型加载失败: {e}")
    raise

# 3. 定义核心的检测接口
@app.post("/detect/")
async def detect_image(
    file: UploadFile = File(...),
    threshold: float = Query(default=0.75, ge=0.0, le=1.0, description="判定为nsfw的置信度阈值（0-1），默认0.75")
):
    """
    接收上传的图片和可选阈值，返回NSFW检测结果。
    - **file**: 图片文件
    - **threshold**: 判定为违规的置信度阈值，范围0~1，默认0.75。nsfw分数高于此值则判定为违规。
    """
    # 检查上传文件类型
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="请上传图片文件")

    try:
        # 读取图片数据
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")  # 确保为RGB格式

        # 使用模型进行预测
        results = classifier(image)

        # 从结果中提取nsfw标签的分数，并进行阈值判断
        nsfw_score = 0.0
        for item in results:
            if item["label"] == "nsfw":
                nsfw_score = item["score"]
                break

        is_unsafe = nsfw_score > threshold
        if is_unsafe:
            hint_message = "检测结果：NSFW"
        else:
            hint_message = "检测结果：SAFE"
        # 格式化返回结果
        return {

            "filename": file.filename,
            "detection_results": results,
            "device_used": "gpu" if device == 0 else "cpu",
            "is_unsafe": is_unsafe,
            "nsfw_score": nsfw_score,
            "threshold_used": threshold,
            "hint_message":hint_message
        }
    except Exception as e:
        logger.error(f"处理图片时出错: {e}")
        raise HTTPException(status_code=500, detail="图片处理失败")

# 4. 一个简单的健康检查端点
@app.get("/")
async def root():
    return {"message": "NSFW图像检测API服务正在运行。请访问 /docs 查看交互式文档。"}
# uvicorn main:app --reload --host 127.0.0.1 --port 9998


import hashlib
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from transformers import pipeline
from PIL import Image
import io
import torch
import logging

# --- 配置日志 ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("NSFW_System")

# 全局变量用于存储模型
ml_models = {}


# --- 1. 生命周期管理 (新版FastAPI推荐) ---
# 这样写比全局执行更规范，能防止并在模型加载失败时优雅退出
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时加载模型
    device = 0 if torch.cuda.is_available() else -1
    logger.info(f"正在初始化系统... 检测到运行设备: {'GPU (CUDA)' if device == 0 else 'CPU'}")

    try:
        # 加载FalconsAI模型
        ml_models["classifier"] = pipeline(
            "image-classification",
            model="Falconsai/nsfw_image_detection",
            device=device
        )
        logger.info("✅ 非法内容检测模型加载完成！系统准备就绪。")
    except Exception as e:
        logger.error(f"❌ 模型加载失败: {e}")
        raise e

    yield  # 运行服务

    # 关闭时清理资源 (如果有)
    ml_models.clear()
    logger.info("系统已关闭，资源已释放。")


app = FastAPI(
    title="基于智能合约的非法内容检测系统后端",
    description="Graduation Thesis: AI Content Detection Module",
    version="1.0.0",
    lifespan=lifespan
)


# --- 辅助函数：计算文件哈希 (用于区块链存证) ---
def calculate_hash(content: bytes) -> str:
    """计算文件的SHA-256哈希值，用于后续上链存证"""
    return hashlib.sha256(content).hexdigest()


# --- 2. 核心检测接口 ---
@app.post("/api/v1/detect", summary="上传图片并检测违规内容")
async def detect_image(
        file: UploadFile = File(..., description="待检测的图片文件"),
        threshold: float = Query(default=0.75, ge=0.0, le=1.0, description="违规判定阈值")
):
    # 1. 格式校验
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="文件格式错误，仅支持图片上传")

    try:
        # 2. 读取文件并计算Hash (关键步骤：为了任务书中的'可验证、不可篡改')
        contents = await file.read()
        file_hash = calculate_hash(contents)

        # 3. 图像预处理
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # 4. 模型推理
        classifier = ml_models["classifier"]
        results = classifier(image)

        # 5. 解析结果
        # Falcons模型通常返回 [{'label': 'nsfw', 'score': 0.9}, {'label': 'normal', 'score': 0.1}]
        nsfw_score = 0.0
        normal_score = 0.0

        for item in results:
            if item["label"] == "nsfw":
                nsfw_score = item["score"]
            elif item["label"] == "normal":
                normal_score = item["score"]

        # 6. 业务逻辑判定
        is_unsafe = nsfw_score > threshold

        # 7. 构建返回数据 (这是给前端或智能合约预言机看的)
        response_data = {
            "file_name": file.filename,
            "content_hash": file_hash,  # 这个字段将来要写进智能合约！
            "is_unsafe": is_unsafe,
            "confidence_score": round(nsfw_score, 4),
            "threshold_used": threshold,
            "detection_details": results,
            "message": "警告：检测到违规内容" if is_unsafe else "通过：内容合规"
        }

        logger.info(f"处理完成: {file.filename} | Unsafe: {is_unsafe} | Score: {nsfw_score:.4f}")
        return response_data

    except Exception as e:
        logger.error(f"处理异常: {e}")
        raise HTTPException(status_code=500, detail=f"内部处理错误: {str(e)}")

# 启动指令: uvicorn main:app --reload --host 127.0.0.1 --port 9998

# - 'eval_loss': 0.07463177293539047,
# - 'eval_accuracy': 0.980375,
# - 'eval_runtime': 304.9846,
# - 'eval_samples_per_second': 52.462,
# - 'eval_steps_per_second': 3.279

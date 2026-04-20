import hashlib
import json
import logging
import io
import torch
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
# 移除 pipeline，替换为底层模型库
from transformers import ViTImageProcessor, ViTForImageClassification
from peft import PeftModel
from PIL import Image
from web3 import Web3
from datetime import datetime
import os
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Form

# --- 配置日志 ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("Graduation_System")



# --- 全局变量 ---

# 图片本地存储目录
UPLOAD_DIR = "./uploaded_images"
os.makedirs(UPLOAD_DIR, exist_ok=True)
ml_models = {}
# 简单的哈希去重（保存在内存里）
processed_hashes = set()

# ==========================================
# 1. 区块链配置 (保持不变)
# ==========================================
CONTRACT_ADDRESS = "0x5FbDB2315678afecb367f032d93F642f64180aa3"
HARDHAT_URL = "http://127.0.0.1:8545"
w3 = Web3(Web3.HTTPProvider(HARDHAT_URL))
# 直接保存为 bytes 对象，供后续调用
HARDHAT_ARTIFACT_PATH = r"D:\AAAWorkData\graduation_project\blockchain\artifacts\contracts\ContentAudit.sol\ContentAudit.json"

def get_contract_abi():
    if not os.path.exists(HARDHAT_ARTIFACT_PATH):
        print(f"❌ 警告: 找不到合约编译文件，请检查路径: {HARDHAT_ARTIFACT_PATH}")
        return []

    with open(HARDHAT_ARTIFACT_PATH, "r", encoding="utf-8") as f:
        artifact = json.load(f)
        return artifact["abi"]

# 2. 直接获取最新的 ABI
CONTRACT_ABI = get_contract_abi()


try:
    if w3.is_connected():
        logger.info(f"✅ 区块链连接成功! 当前区块高度: {w3.eth.block_number}")
        w3.eth.default_account = w3.eth.accounts[0]
        contract = w3.eth.contract(address=CONTRACT_ADDRESS, abi=CONTRACT_ABI)
    else:
        logger.warning("❌ 无法连接到区块链")
        contract = None
except Exception as e:
    logger.error(f"区块链初始化异常: {e}")
    contract = None


# ==========================================
# 2. 辅助函数 (保持不变)
# ==========================================
def calculate_hash(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def upload_to_blockchain(file_hash: str, is_unsafe: bool, score: float,submitter_address: str):
    if not contract: return "Blockchain_Disconnected"
    try:
        score_int = int(score * 10000)
        tx_hash = contract.functions.logDetection(file_hash, is_unsafe, score_int,Web3.to_checksum_address(submitter_address)).transact()
        receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
        logger.info(f"🔗 上链成功! 交易哈希: {receipt.transactionHash.hex()}")
        return receipt.transactionHash.hex()
    except Exception as e:
        if "Record already exists" in str(e): return "Already_On_Chain"
        return f"Error: {str(e)}"


# ==========================================
# 3. FastAPI 生命周期与接口 (集成自定义 LoRA 模型)
# ==========================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 自动识别 RTX 3060 或 CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 设定路径：指向你的底座和微调权重
    base_model_name = "FalconsAI/nsfw_image_detection"
    lora_model_path = "./final_falconsai_lora"

    try:
        logger.info("⏳ 正在组装自定义二次元检测引擎...")

        # 1. 加载处理器 (负责缩放图片)
        processor = ViTImageProcessor.from_pretrained(lora_model_path)

        # 2. 加载原版底座
        base_model = ViTForImageClassification.from_pretrained(base_model_name)

        # 3. 注入 LoRA 灵魂，并推送到显卡
        model = PeftModel.from_pretrained(base_model, lora_model_path)
        model.to(device)
        model.eval()  # 开启推理模式

        # 保存到全局变量供接口调用
        ml_models["processor"] = processor
        ml_models["model"] = model
        ml_models["device"] = device

        logger.info(f"✅ AI 自定义模型加载完成！运行设备: {device}")
    except Exception as e:
        logger.error(f"模型加载失败，请检查路径 './final_falconsai_lora' 是否存在: {e}")

    # 权限自检逻辑
    ORACLE_ROLE_HASH = w3.keccak(text="ORACLE_ROLE")
    has_perm = contract.functions.hasRole(ORACLE_ROLE_HASH, w3.eth.default_account).call()

    if has_perm:
        logger.info(f"🔑 权限校验通过！后端地址{w3.eth.default_account}已获得ORACLE_ROLE授权")
    else:
        logger.error(f"❌ 权限校验失败！后端地址 {w3.eth.default_account} 未获得ORACLE_ROLE授权")

    yield
    ml_models.clear()


app = FastAPI(title="智能合约非法内容检测系统", version="Final.1", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/v1/detect", summary="上传图片-自定义AI检测-自动上链")
async def detect_image(file: UploadFile = File(...), user_address: str = Form(...), threshold: float = Query(0.75)):
    contents = await file.read()
    file_hash = calculate_hash(contents)

    # ==========================================
    # 核心防御逻辑：幂等性校验 (防刷单/省 Gas)
    # ==========================================
    if not contract:
        raise HTTPException(status_code=500, detail="区块链连接未就绪")

    try:
        # 直接调用合约的 records 映射查询
        # data 结构: [contentHash, isUnsafe, aiScore, timestamp, auditor, submitter]
        existing_data = contract.functions.records(file_hash).call()

        # 检查时间戳是否不为 0 (如果不为 0，说明链上已有记录)
        if existing_data[3] != 0:
            logger.info(f"🎯 命中链上缓存! Hash: {file_hash} | 跳过 AI 推理与交易")
            return {
                "file_name": file.filename,
                "file_hash": file_hash,
                "image_url": f"http://127.0.0.1:8001/api/v1/image/{file_hash}",
                "is_unsafe": existing_data[1],
                "confidence_score": existing_data[2] / 10000,
                "blockchain_status": {
                    "transaction_hash": "ALREADY_ON_CHAIN",
                    "status": "Success",
                    "note": "检测到历史存证，已自动调取链上数据"
                }
            }
    except Exception as e:
        logger.error(f"查询历史记录时发生异常: {e}")
        # 这里可以选择继续跑 AI，也可以选择报错，建议继续跑 AI 以保证可用性

    # ==========================================
    # 如果链上没有记录，才执行以下昂贵操作
    # ==========================================

    # ---- 1. 保存图片到本地 (存储去重) ----
    file_ext = os.path.splitext(file.filename)[1] or ".jpg"
    local_filename = f"{file_hash}{file_ext}"
    local_path = os.path.join(UPLOAD_DIR, local_filename)

    # 只有本地不存在时才写入，进一步减少 IO 消耗
    if not os.path.exists(local_path):
        with open(local_path, "wb") as f:
            f.write(contents)

    # ---- 2. AI 推理 (昂贵操作) ----
    processor = ml_models["processor"]
    model = ml_models["model"]
    device = ml_models["device"]

    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
        nsfw_score = probabilities[1].item()
    except Exception as e:
        logger.error(f"推理过程中发生错误: {e}")
        raise HTTPException(status_code=500, detail="模型推理失败")

    is_unsafe = nsfw_score > threshold

    # ---- 3. 上链存证 (消耗 Gas) ----
    tx_hash = upload_to_blockchain(file_hash, is_unsafe, nsfw_score, user_address)
    formatted_tx_hash = tx_hash if tx_hash.startswith("0x") else f"0x{tx_hash}"

    return {
        "file_name": file.filename,
        "file_hash": file_hash,
        "image_url": f"http://127.0.0.1:8001/api/v1/image/{file_hash}",
        "is_unsafe": is_unsafe,
        "confidence_score": round(nsfw_score, 4),
        "blockchain_status": {
            "transaction_hash": formatted_tx_hash,
            "status": "Success" if "0x" in formatted_tx_hash.lower() else formatted_tx_hash
        }
    }


@app.get("/api/v1/record/{file_hash}", summary="通过图片哈希查询链上记录")
async def get_record(file_hash: str):
    # (查询逻辑保持不变)
    if not contract:
        raise HTTPException(status_code=500, detail="区块链连接未就绪")

    try:
        data = contract.functions.records(file_hash).call()
        if data[3] == 0:
            raise HTTPException(status_code=404, detail="链上未找到该图片的检测记录")

        return {
            "status": "Success",
            "source": "Blockchain",
            "record": {
                "content_hash": data[0],
                "is_unsafe": data[1],
                "ai_score": data[2] / 10000,
                "detect_time": datetime.fromtimestamp(data[3]).strftime('%Y-%m-%d %H:%M:%S'),
                "auditor_address": data[4],
                "submitter_address": data[5]
            }
        }
    except Exception as e:
        logger.error(f"查询失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/image/{file_hash}", summary="下载原始图片用于审计")
async def get_image(file_hash: str):
    logger.info(f"🔍 正在查找哈希为 {file_hash} 的图片...")
    for filename in os.listdir(UPLOAD_DIR):
        # 确保完全匹配哈希部分（排除掉扩展名后的对比）
        if filename.split('.')[0] == file_hash:
            file_path = os.path.join(UPLOAD_DIR, filename)
            logger.info(f"✅ 找到图片: {file_path}")
            return FileResponse(file_path)  # 让 FastAPI 自动判断 mime_type
    logger.warning(f"❌ 未找到图片: {file_hash}")
    raise HTTPException(status_code=404, detail="Image not found")

# npx hardhat node
# npx hardhat ignition deploy D:\AAAWorkData\graduation_project\blockchain\ignition\modules\ContentAudit.ts --network localhost
# 启动命令: uvicorn main_abr2024_in_falconsai:app --reload --host 127.0.0.1 --port 8001


import hashlib
import json
import logging
import io
import torch
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from transformers import pipeline
from PIL import Image
from web3 import Web3
from datetime import datetime  # 新增：用于处理时间戳

# --- 配置日志 ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("Graduation_System")

# --- 全局变量 ---
ml_models = {}

# ==========================================
# 1. 区块链配置
# ==========================================
CONTRACT_ADDRESS = "0x5FbDB2315678afecb367f032d93F642f64180aa3"
HARDHAT_URL = "http://127.0.0.1:8545"
w3 = Web3(Web3.HTTPProvider(HARDHAT_URL))

# 合约接口定义 (ABI) - 增加了查询所需的 records 定义
CONTRACT_ABI = [
    {
        "inputs": [
            {"internalType": "string", "name": "_contentHash", "type": "string"},
            {"internalType": "bool", "name": "_isUnsafe", "type": "bool"},
            {"internalType": "uint256", "name": "_score", "type": "uint256"}
        ],
        "name": "logDetection",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function"
    },
    {
        "inputs": [{"internalType": "string", "name": "", "type": "string"}],
        "name": "records",
        "outputs": [
            {"internalType": "string", "name": "contentHash", "type": "string"},
            {"internalType": "bool", "name": "isUnsafe", "type": "bool"},
            {"internalType": "uint256", "name": "score", "type": "uint256"},
            {"internalType": "uint256", "name": "timestamp", "type": "uint256"},
            {"internalType": "address", "name": "auditor", "type": "address"}
        ],
        "stateMutability": "view",
        "type": "function"
    }
]

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
# 2. 辅助函数
# ==========================================
def calculate_hash(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()

def upload_to_blockchain(file_hash: str, is_unsafe: bool, score: float):
    if not contract: return "Blockchain_Disconnected"
    try:
        score_int = int(score * 10000)
        tx_hash = contract.functions.logDetection(file_hash, is_unsafe, score_int).transact()
        receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
        logger.info(f"🔗 上链成功! 交易哈希: {receipt.transactionHash.hex()}")
        return receipt.transactionHash.hex()
    except Exception as e:
        if "Record already exists" in str(e): return "Already_On_Chain"
        return f"Error: {str(e)}"

# ==========================================
# 3. FastAPI 生命周期与接口
# ==========================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    device = 0 if torch.cuda.is_available() else -1
    try:
        ml_models["classifier"] = pipeline("image-classification", model="Falconsai/nsfw_image_detection", device=device)
        logger.info("✅ AI 模型加载完成！")
    except Exception as e:
        logger.error(f"模型加载失败: {e}")
    yield
    ml_models.clear()

app = FastAPI(title="智能合约非法内容检测系统", version="Final.0", lifespan=lifespan)

@app.post("/api/v1/detect", summary="上传图片-AI检测-自动上链")
async def detect_image(file: UploadFile = File(...), threshold: float = Query(0.75)):
    contents = await file.read()
    file_hash = calculate_hash(contents)

    image = Image.open(io.BytesIO(contents)).convert("RGB")
    classifier = ml_models["classifier"]
    results = classifier(image)

    nsfw_score = next((item["score"] for item in results if item["label"] == "nsfw"), 0.0)
    is_unsafe = nsfw_score > threshold

    logger.info(f"处理图片: {file.filename} | Hash: {file_hash}")
    tx_hash = upload_to_blockchain(file_hash, is_unsafe, nsfw_score)

    return {
        "file_name": file.filename,
        "file_hash": file_hash,  # <-- 显式返回图片哈希
        "is_unsafe": is_unsafe,
        "confidence_score": round(nsfw_score, 4),
        "blockchain_status": {
            "transaction_hash": tx_hash,
            "status": "Success" if "0x" in tx_hash else tx_hash
        }
    }

@app.get("/api/v1/record/{file_hash}", summary="通过图片哈希查询链上记录")
async def get_record(file_hash: str):
    """
    输入图片的 SHA-256 哈希值，从区块链实时调取存证数据
    """
    if not contract:
        raise HTTPException(status_code=500, detail="区块链连接未就绪")

    try:
        # 调用合约的只读 mapping
        # 返回顺序由合约 struct 决定：contentHash, isUnsafe, score, timestamp, auditor
        data = contract.functions.records(file_hash).call()

        # 如果 timestamp (data[3]) 为 0，说明该哈希从未被存证
        if data[3] == 0:
            raise HTTPException(status_code=404, detail="链上未找到该图片的检测记录")

        return {
            "status": "Success",
            "source": "Blockchain",
            "record": {
                "content_hash": data[0],
                "is_unsafe": data[1],
                "ai_score": data[2] / 10000,  # 还原回小数
                "detect_time": datetime.fromtimestamp(data[3]).strftime('%Y-%m-%d %H:%M:%S'),
                "auditor_address": data[4]
            }
        }
    except Exception as e:
        logger.error(f"查询失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# 启动命令: uvicorn main:app --reload --host 127.0.0.1 --port 8000
# npx hardhat node
# npx hardhat ignition deploy D:\AAAWorkData\graduation_project\blockchain\ignition\modules\ContentAudit.ts --network localhost

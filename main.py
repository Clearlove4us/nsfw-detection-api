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

# --- 配置日志 ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("Graduation_System")

# --- 全局变量 ---
ml_models = {}

# ==========================================
# 1. 区块链配置 (核心部分)
# ==========================================
# 合约地址 (刚刚部署得到的)
CONTRACT_ADDRESS = "0x5FbDB2315678afecb367f032d93F642f64180aa3"

# 连接到本地 Hardhat 节点
HARDHAT_URL = "http://127.0.0.1:8545"
w3 = Web3(Web3.HTTPProvider(HARDHAT_URL))

# 合约接口定义 (ABI)

CONTRACT_ABI = [
    {
        "inputs": [{"internalType": "string", "name": "_contentHash", "type": "string"}],
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

# 初始化合约对象
try:
    if w3.is_connected():
        logger.info(f"✅ 区块链连接成功! 当前区块高度: {w3.eth.block_number}")
        # 设置默认发送账户 (使用 Hardhat 的 Account #0)
        w3.eth.default_account = w3.eth.accounts[0]
        contract = w3.eth.contract(address=CONTRACT_ADDRESS, abi=CONTRACT_ABI)
    else:
        logger.warning("❌ 无法连接到区块链，请确保 'npx hardhat node' 正在运行")
        contract = None
except Exception as e:
    logger.error(f"区块链初始化异常: {e}")
    contract = None


# ==========================================
# 2. 辅助函数
# ==========================================
def calculate_hash(content: bytes) -> str:
    """计算文件的SHA-256哈希值"""
    return hashlib.sha256(content).hexdigest()


def upload_to_blockchain(file_hash: str, is_unsafe: bool, score: float):
    """将检测结果写入智能合约"""
    if not contract:
        return "Blockchain_Disconnected"

    try:
        # Solidity 不支持小数，我们将 0.9856 转换为 9856 (放大1万倍)
        score_int = int(score * 10000)

        # 发送交易
        # 注意：在本地 Hardhat 节点，default_account 是自动解锁的，可以直接 transact
        tx_hash = contract.functions.logDetection(
            file_hash,
            is_unsafe,
            score_int
        ).transact()

        # 等待交易被打包确认
        receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
        logger.info(f"🔗 上链成功! 交易哈希: {receipt.transactionHash.hex()}")
        return receipt.transactionHash.hex()

    except Exception as e:
        # 如果是重复上传（哈希相同），合约会报错，这是正常的逻辑
        if "Record already exists" in str(e):
            logger.warning("该文件已存在于链上，跳过存证。")
            return "Already_On_Chain"
        logger.error(f"❌ 上链失败: {e}")
        return f"Error: {str(e)}"


# ==========================================
# 3. FastAPI 生命周期与接口
# ==========================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时加载模型
    device = 0 if torch.cuda.is_available() else -1
    logger.info(f"正在加载 AI 模型... 设备: {'GPU' if device == 0 else 'CPU'}")

    try:
        ml_models["classifier"] = pipeline(
            "image-classification",
            model="Falconsai/nsfw_image_detection",
            device=device
        )
        logger.info("✅ FalconsAI 模型加载完成！")
    except Exception as e:
        logger.error(f"模型加载失败: {e}")

    yield
    ml_models.clear()


app = FastAPI(title="智能合约非法内容检测系统", version="Final.0", lifespan=lifespan)


@app.post("/api/v1/detect", summary="上传图片-AI检测-自动上链")
async def detect_image(
        file: UploadFile = File(...),
        threshold: float = Query(0.75, description="违规判定阈值")
):
    # 1. 读取与哈希
    contents = await file.read()
    file_hash = calculate_hash(contents)

    # 2. AI 检测
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    classifier = ml_models["classifier"]
    results = classifier(image)

    # 解析分数
    nsfw_score = 0.0
    for item in results:
        if item["label"] == "nsfw":
            nsfw_score = item["score"]
            break

    is_unsafe = nsfw_score > threshold

    # 3. 区块链存证 (核心任务点)
    # 只有当系统判定完成，且区块链连接正常时，才执行上链
    logger.info(f"开始处理: {file.filename} | Hash: {file_hash[:8]}...")
    tx_hash = upload_to_blockchain(file_hash, is_unsafe, nsfw_score)

    # 4. 返回结果
    return {
        "file_name": file.filename,
        "is_unsafe": is_unsafe,
        "confidence_score": round(nsfw_score, 4),
        "blockchain_status": {
            "stored_on_chain": tx_hash not in ["Blockchain_Disconnected", "Error"],
            "transaction_hash": tx_hash,
            "note": "交易哈希可用于在区块链浏览器中查证"
        },
        "system_message": "检测完成并已尝试存证"
    }


# 启动命令: uvicorn main:app --reload --host 127.0.0.1 --port 8000
# npx hardhat node
# npx hardhat ignition deploy D:\AAAWorkData\graduation_project\blockchain\ignition\modules\ContentAudit.ts --network localhost

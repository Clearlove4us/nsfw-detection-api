<template>
  <div class="app-container">
    <header class="header">
      <div class="logo">🛡️ 智能合约内容审计系统 <el-tag size="small" type="info">v1.0</el-tag></div>
      <div class="wallet-info">
        <el-tag v-if="account" type="success" effect="dark">
          {{ shortAccount }} | {{ balance }} ETH
          <el-icon style="margin-left: 8px; cursor: pointer;" @click="refreshBalance">
            <Refresh />
          </el-icon>
        </el-tag>
        <el-button v-else type="primary" @click="connectWallet">连接 MetaMask</el-button>
      </div>
    </header>

    <el-main>
      <el-row :gutter="20">
        <el-col :span="12">
          <el-card class="main-card">
            <template #header><b>📸 图像安全审计</b></template>
            
            <el-upload
              class="upload-area"
              drag
              action="http://127.0.0.1:8001/api/v1/detect"
              :on-success="handleUploadSuccess"
              :on-error="handleUploadError"
              :before-upload="beforeUpload"
              :data="{ user_address: account }"
              name="file"
            >
              <el-icon class="el-icon--upload"><upload-filled /></el-icon>
              <div class="el-upload__text">拖拽文件或 <em>点击上传进行 AI 检测</em></div>
            </el-upload>

            <transition name="el-fade-in">
              <div v-if="auditResult" class="result-box">
                <el-result
                  :icon="auditResult.is_unsafe ? 'error' : 'success'"
                  :title="auditResult.is_unsafe ? '发现违规内容' : '内容安全'"
                  :sub-title="'AI 置信度: ' + (auditResult.confidence_score * 100).toFixed(2) + '%'"
                >
                  <template #extra>
                    <p class="hash-text">文件哈希: {{ auditResult.file_hash }}</p>
                    <el-link :href="'https://etherscan.io/tx/' + auditResult.blockchain_status.transaction_hash" target="_blank" type="primary">
                      🔗 链上交易凭证 (查看交易)
                    </el-link>
                  </template>
                </el-result>
              </div>
            </transition>
          </el-card>
        </el-col>

        <el-col :span="12">
          <el-card class="main-card">
            <template #header><b>🔍 链上存证溯源</b></template>
            <div class="search-box">
              <el-input v-model="searchHash" placeholder="输入图片哈希值查询链上原始记录">
                <template #append>
                  <el-button @click="queryBlockchain">查询</el-button>
                </template>
              </el-input>
            </div>

            <div v-if="chainRecord" class="history-list">
              <el-descriptions title="区块链原始存证信息" :column="1" border>
                <el-descriptions-item label="存证内容快照">
                  <el-image
                    style="width: 100px; height: 100px; border-radius: 8px;"
                    :src="`http://127.0.0.1:8001/api/v1/image/${currentFileHash}`"
                    fit="cover"
                    :preview-src-list="[`http://127.0.0.1:8001/api/v1/image/${currentFileHash}`]"
                  />
                </el-descriptions-item>
                <el-descriptions-item label="存证状态"><el-tag type="success">永久锁定</el-tag></el-descriptions-item>
                <el-descriptions-item label="AI 判定结果">
                  {{ chainRecord.is_unsafe ? '🔴 违规' : '🟢 正常' }}
                </el-descriptions-item>
                <el-descriptions-item label="链上记录得分">{{ chainRecord.ai_score }}</el-descriptions-item>
                <el-descriptions-item label="审计时间">{{ chainRecord.detect_time }}</el-descriptions-item>
                <el-descriptions-item label="审计员地址 (Oracle)">
                  <span class="addr">{{ chainRecord.auditor_address }}</span>
                </el-descriptions-item>
                <el-descriptions-item label="内容上传者 (User)">
                  <span class="addr" style="color: #67c23a;">{{ chainRecord.submitter_address }}</span>
                </el-descriptions-item>
              </el-descriptions>
            </div>
            <el-empty v-else description="暂无查询结果" />
          </el-card>
        </el-col>
      </el-row>
    </el-main>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, onUnmounted } from 'vue'
import { ElMessage, ElLoading } from 'element-plus'
import { UploadFilled, Refresh } from '@element-plus/icons-vue'
import { ethers } from 'ethers'
import axios from 'axios'

const account = ref('')
const balance = ref('0.00')
const auditResult = ref(null)
const searchHash = ref('')
const chainRecord = ref(null)
const currentFileHash = ref('')

// 缩写地址
const shortAccount = computed(() => {
  return account.value ? `${account.value.slice(0, 6)}...${account.value.slice(-4)}` : ''
})

// ---------- 核心网络检查函数（修复重复切换错误）----------
async function ensureHardhatNetwork() {
  if (!window.ethereum) return false
  const targetChainId = '0x7a69'  // 31337
  const currentChainId = await window.ethereum.request({ method: 'eth_chainId' })
  
  // 已经在正确网络，直接返回成功
  if (currentChainId === targetChainId) return true

  // 尝试切换
  try {
    await window.ethereum.request({
      method: 'wallet_switchEthereumChain',
      params: [{ chainId: targetChainId }],
    })
    return true
  } catch (switchError) {
    // 如果目标网络未添加，则添加
    if (switchError.code === 4902) {
      try {
        await window.ethereum.request({
          method: 'wallet_addEthereumChain',
          params: [{
            chainId: targetChainId,
            chainName: 'Hardhat Local Network',
            rpcUrls: ['http://127.0.0.1:8545'],
            nativeCurrency: { name: 'ETH', symbol: 'ETH', decimals: 18 }
          }]
        })
        return true
      } catch (addError) {
        console.error('添加 Hardhat 网络失败', addError)
        return false
      }
    }
    console.error('切换网络失败', switchError)
    return false
  }
}

// ---------- 余额同步函数（优化版）----------
async function updateBalance() {
  if (!window.ethereum || !account.value) return
  
  // 1. 确保网络正确（不会重复切换导致报错）
  const isCorrectNetwork = await ensureHardhatNetwork()
  if (!isCorrectNetwork) {
    ElMessage.warning('请手动切换到 Hardhat 本地网络')
    return
  }

  // 2. 获取余额
  try {
    const rawBalance = await window.ethereum.request({
      method: 'eth_getBalance',
      params: [account.value, 'latest']
    })
    const ethValue = ethers.formatEther(rawBalance)
    balance.value = parseFloat(ethValue).toFixed(4)
    console.log("✅ 余额已更新:", balance.value)
  } catch (err) {
    console.error("❌ 获取余额失败:", err)
    ElMessage.error('获取余额失败，请检查网络连接')
  }
}

// 手动刷新余额（方案一核心按钮）
const refreshBalance = async () => {
  await updateBalance()
  ElMessage.success('余额已刷新')
}

// ---------- 初始化与事件监听 ----------
onMounted(async () => {
  if (window.ethereum) {
    try {
      // 获取当前账户
      const accounts = await window.ethereum.request({ method: 'eth_accounts' })
      if (accounts.length > 0) {
        account.value = accounts[0]
        await updateBalance()
        console.log("🚀 初始化完成，当前账号:", account.value)
      }

      // 监听账户切换
      window.ethereum.on('accountsChanged', async (newAccounts) => {
        if (newAccounts.length > 0) {
          account.value = newAccounts[0]
          await updateBalance()
          ElMessage.info('已切换至新账号')
        } else {
          account.value = ''
          balance.value = '0.00'
          ElMessage.warning('钱包连接已断开')
        }
      })

      // 监听网络切换（自动刷新页面以重新加载）
      window.ethereum.on('chainChanged', () => {
        window.location.reload()
      })
    } catch (e) {
      console.error("初始化钱包监听失败:", e)
    }
  }
})

// 可选：组件卸载时清理监听（全局监听一般不需要，但为了规范）
onUnmounted(() => {
  if (window.ethereum) {
    window.ethereum.removeAllListeners?.()
  }
})

// ---------- 业务函数 ----------
async function connectWallet() {
  if (!window.ethereum) return ElMessage.error('请安装 MetaMask!')
  try {
    const accounts = await window.ethereum.request({ method: 'eth_requestAccounts' })
    account.value = accounts[0]
    await updateBalance()
    ElMessage.success('钱包已连接')
  } catch (err) {
    ElMessage.error('连接被拒绝')
  }
}

function beforeUpload() {
  if (!account.value) {
    ElMessage.warning('请先连接钱包以同步审计身份')
    return false
  }
  return true
}

async function handleUploadSuccess(res) {
  auditResult.value = res
  currentFileHash.value = res.file_hash
  searchHash.value = res.file_hash
  await queryBlockchain()      // 自动查询右侧记录
  await updateBalance()        // 上传后刷新余额（可能有 gas 消耗）
  ElMessage.success('审计完成，存证已上链！')
}

function handleUploadError() {
  ElMessage.error('上传失败，请检查后端服务是否启动')
}

async function queryBlockchain() {
  if (!searchHash.value) return
  const loading = ElLoading.service({ text: '正在调取区块链数据...' })
  try {
    const res = await axios.get(`http://127.0.0.1:8001/api/v1/record/${searchHash.value}`)
    chainRecord.value = res.data.record
    currentFileHash.value = searchHash.value
    loading.close()
  } catch (err) {
    loading.close()
    chainRecord.value = null
    currentFileHash.value = ''
    ElMessage.error('未找到链上记录，请确认哈希值正确')
  }
}
</script>

<style scoped>
.app-container { min-height: 100vh; background-color: #f5f7fa; }
.header { background: #1a1a1a; color: white; padding: 0 40px; height: 60px; display: flex; align-items: center; justify-content: space-between; box-shadow: 0 2px 10px rgba(0,0,0,0.2); }
.logo { font-size: 1.2rem; font-weight: bold; }
.main-card { margin-top: 20px; min-height: 500px; border-radius: 12px; }
.upload-area { margin: 20px 0; }
.result-box { margin-top: 20px; padding: 20px; background: #fff; border-radius: 8px; }
.hash-text { font-size: 12px; color: #909399; word-break: break-all; margin: 10px 0; }
.search-box { margin-bottom: 30px; }
.history-list { animation: fadeInUp 0.5s; }
.addr { font-size: 10px; color: #409eff; }

@keyframes fadeInUp {
  from { opacity: 0; transform: translateY(20px); }
  to { opacity: 1; transform: translateY(0); }
}
</style>

//npm run dev

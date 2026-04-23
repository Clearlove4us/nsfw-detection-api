<template>
  <div class="app-container">
    <header class="header">
      <div class="logo">🛡️ 智能合约内容审计系统 <el-tag size="small" type="info">v2.0 全透明审计版</el-tag></div>
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

    <el-main class="main-content">
      <el-tabs v-model="activeTab" class="custom-tabs" @tab-click="handleTabChange">
        
        <el-tab-pane label="🛠️ 审计工作台" name="workspace">
          <el-row :gutter="20">
            <el-col :span="12">
              <el-card class="main-card shadow-card">
                <template #header><b>📸 图像安全检测上链</b></template>
                
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
                          🔗 链上交易凭证
                        </el-link>
                      </template>
                    </el-result>
                  </div>
                </transition>
              </el-card>
            </el-col>

            <el-col :span="12">
              <el-card class="main-card shadow-card">
                <template #header><b>🔍 链上存证溯源</b></template>
                <div class="search-box">
                  <el-input v-model="searchHash" placeholder="输入图片哈希值查询链上记录">
                    <template #append>
                      <el-button @click="queryBlockchain">去区块链查询</el-button>
                    </template>
                  </el-input>
                </div>

                <div v-if="chainRecord" class="history-list">
                  <el-descriptions title="区块链原始存证信息" :column="1" border>
                    <el-descriptions-item label="存证内容快照">
                      <div v-if="chainRecord.is_unsafe" class="unsafe-badge-small">
                        已依规拦截
                      </div>
                      <el-image
                        v-else
                        style="width: 100px; height: 100px; border-radius: 8px;"
                        :src="`http://127.0.0.1:8001/api/v1/image/${currentFileHash}`"
                        fit="cover"
                        :preview-src-list="[`http://127.0.0.1:8001/api/v1/image/${currentFileHash}`]"
                      />
                    </el-descriptions-item>
                    <el-descriptions-item label="AI 判定结果">
                      <el-tag :type="chainRecord.is_unsafe ? 'danger' : 'success'" effect="dark">
                        {{ chainRecord.is_unsafe ? '🔴 违规内容' : '🟢 合规内容' }}
                      </el-tag>
                    </el-descriptions-item>
                    <el-descriptions-item label="链上哈希索引">{{ chainRecord.content_hash }}</el-descriptions-item>
                    <el-descriptions-item label="审计时间">{{ chainRecord.detect_time }}</el-descriptions-item>
                    <el-descriptions-item label="Oracle 地址">
                      <span class="addr">{{ chainRecord.auditor_address }}</span>
                    </el-descriptions-item>
                    <el-descriptions-item label="内容上传者 (User)">
                      <span class="addr" style="color: #67c23a;">{{ chainRecord.submitter_address }}</span>
                    </el-descriptions-item>
                  </el-descriptions>
                </div>
                <el-empty v-else description="输入哈希或从大厅选择记录进行核验" />
              </el-card>
            </el-col>
          </el-row>
        </el-tab-pane>

        <el-tab-pane label="🌐 公开审计大厅" name="hall">
          <div class="hall-header">
            <h3>社区存证公示墙</h3>
            <p class="subtitle">任何人均可在此查看系统处理过的公开内容，并一键核对区块链存证结果。</p>
          </div>
          
          <div v-loading="loadingRecords" style="min-height: 300px;">
            <el-row :gutter="20">
              <el-col :span="6" v-for="(item, index) in recordsList" :key="index" style="margin-bottom: 20px;">
                <el-card :body-style="{ padding: '0px' }" class="gallery-card">
                  <div class="image-wrapper">
                    <div v-if="item.is_unsafe" class="unsafe-mask">
                      <el-icon size="40"><Warning /></el-icon>
                      <span>内容违规 已拦截</span>
                    </div>
                    <el-image
                      v-else
                      :src="`http://127.0.0.1:8001/api/v1/image/${item.file_hash}`"
                      class="gallery-image"
                      fit="cover"
                      loading="lazy"
                    />
                  </div>
                  
                  <div class="card-footer">
                    <div class="hash-tag">哈希: {{ item.file_hash.slice(0, 10) }}...</div>
                    <div class="status-row">
                      <el-tag size="small" :type="item.is_unsafe ? 'danger' : 'success'">
                        {{ item.is_unsafe ? '违规' : '安全' }}
                      </el-tag>
                      <span class="time-text">{{ item.time.split(' ')[0] }}</span>
                    </div>
                    <el-button type="primary" plain size="small" style="width: 100%; margin-top: 10px;" @click="verifyRecord(item.file_hash)">
                      查证链上数据
                    </el-button>
                  </div>
                </el-card>
              </el-col>
            </el-row>
            <el-empty v-if="recordsList.length === 0 && !loadingRecords" description="暂无公示记录" />
          </div>
        </el-tab-pane>

      </el-tabs>
    </el-main>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, onUnmounted } from 'vue'
import { ElMessage, ElLoading } from 'element-plus'
import { UploadFilled, Refresh, Warning } from '@element-plus/icons-vue'
import { ethers } from 'ethers'
import axios from 'axios'

// 基础状态
const account = ref('')
const balance = ref('0.00')
const activeTab = ref('workspace') // 控制当前显示的页面

// 工作台状态
const auditResult = ref(null)
const searchHash = ref('')
const chainRecord = ref(null)
const currentFileHash = ref('')

// 大厅状态
const recordsList = ref([])
const loadingRecords = ref(false)

// 缩写地址
const shortAccount = computed(() => {
  return account.value ? `${account.value.slice(0, 6)}...${account.value.slice(-4)}` : ''
})

// ---------- 区块链钱包逻辑 (保持不变) ----------
async function ensureHardhatNetwork() {
  if (!window.ethereum) return false
  const targetChainId = '0x7a69'
  const currentChainId = await window.ethereum.request({ method: 'eth_chainId' })
  if (currentChainId === targetChainId) return true
  try {
    await window.ethereum.request({ method: 'wallet_switchEthereumChain', params: [{ chainId: targetChainId }] })
    return true
  } catch (e) { return false }
}

async function updateBalance() {
  if (!window.ethereum || !account.value) return
  const isCorrect = await ensureHardhatNetwork()
  if (!isCorrect) return
  try {
    const rawBalance = await window.ethereum.request({ method: 'eth_getBalance', params: [account.value, 'latest'] })
    balance.value = parseFloat(ethers.formatEther(rawBalance)).toFixed(4)
  } catch (err) { console.error(err) }
}

const refreshBalance = async () => {
  await updateBalance()
  ElMessage.success('余额已刷新')
}

async function connectWallet() {
  if (!window.ethereum) return ElMessage.error('请安装 MetaMask!')
  try {
    const accounts = await window.ethereum.request({ method: 'eth_requestAccounts' })
    account.value = accounts[0]
    await updateBalance()
    ElMessage.success('钱包已连接')
  } catch (err) { ElMessage.error('连接被拒绝') }
}

onMounted(async () => {
  if (window.ethereum) {
    const accounts = await window.ethereum.request({ method: 'eth_accounts' })
    if (accounts.length > 0) {
      account.value = accounts[0]
      await updateBalance()
    }
    window.ethereum.on('accountsChanged', (newAcc) => {
      account.value = newAcc[0] || ''
      updateBalance()
    })
  }
})

// ---------- 业务逻辑：上传与查询 ----------
function beforeUpload() {
  if (!account.value) {
    ElMessage.warning('请先连接钱包')
    return false
  }
  return true
}

async function handleUploadSuccess(res) {
  auditResult.value = res
  // 注意：这里已经修改为使用 file_hash
  currentFileHash.value = res.file_hash
  searchHash.value = res.file_hash
  await queryBlockchain()
  await updateBalance()
  ElMessage.success('操作成功！')
}

function handleUploadError() {
  ElMessage.error('服务连接失败')
}

async function queryBlockchain() {
  if (!searchHash.value) return
  const loading = ElLoading.service({ text: '正在调取区块链智能合约...' })
  try {
    const res = await axios.get(`http://127.0.0.1:8001/api/v1/record/${searchHash.value}`)
    chainRecord.value = res.data.record
    currentFileHash.value = searchHash.value
    loading.close()
  } catch (err) {
    loading.close()
    chainRecord.value = null
    currentFileHash.value = ''
    ElMessage.error('链上未找到该记录')
  }
}

// ---------- 业务逻辑：审计大厅 ----------
async function fetchRecords() {
  loadingRecords.value = true
  try {
    const res = await axios.get('http://127.0.0.1:8001/api/v1/records')
    recordsList.value = res.data.data
  } catch (err) {
    ElMessage.error('获取公开记录失败')
  } finally {
    loadingRecords.value = false
  }
}

function handleTabChange(tab) {
  // 当用户切换到大厅时，主动拉取最新数据
  if (tab.paneName === 'hall') {
    fetchRecords()
  }
}

function verifyRecord(hash) {
  // 用户在大厅点击“查证”，跳转回工作台并自动查询
  searchHash.value = hash
  activeTab.value = 'workspace'
  queryBlockchain()
  ElMessage.success('已切换至工作台，正在发起链上核验')
}
</script>

<style scoped>
.app-container { min-height: 100vh; background-color: #f5f7fa; }
.header { background: #1a1a1a; color: white; padding: 0 40px; height: 60px; display: flex; align-items: center; justify-content: space-between; box-shadow: 0 2px 10px rgba(0,0,0,0.2); z-index: 10; position: relative;}
.logo { font-size: 1.2rem; font-weight: bold; }
.main-content { padding: 30px; max-width: 1400px; margin: 0 auto; }
.shadow-card { border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.05); border: none; min-height: 550px;}

.custom-tabs :deep(.el-tabs__item) { font-size: 16px; font-weight: bold; }

/* 审计大厅样式 */
.hall-header { text-align: center; margin-bottom: 30px; }
.hall-header h3 { font-size: 24px; color: #303133; margin-bottom: 10px; }
.subtitle { color: #909399; }

.gallery-card { border-radius: 10px; overflow: hidden; transition: transform 0.3s; }
.gallery-card:hover { transform: translateY(-5px); box-shadow: 0 8px 16px rgba(0,0,0,0.1); }
.image-wrapper { height: 180px; width: 100%; position: relative; background: #f0f2f5;}
.gallery-image { width: 100%; height: 100%; }

/* 违规打码掩码 */
.unsafe-mask {
  position: absolute; top: 0; left: 0; right: 0; bottom: 0;
  background: rgba(40, 40, 40, 0.9);
  display: flex; flex-direction: column; align-items: center; justify-content: center;
  color: #f56c6c; font-weight: bold; font-size: 14px;
}
.unsafe-badge-small {
  width: 100px; height: 100px; border-radius: 8px; background: #ffe6e6; color: #f56c6c;
  display: flex; align-items: center; justify-content: center; font-size: 12px; font-weight: bold; border: 1px dashed #f56c6c;
}

.card-footer { padding: 15px; background: #fff; }
.hash-tag { font-family: monospace; font-size: 12px; color: #606266; margin-bottom: 8px; background: #f4f4f5; padding: 4px; border-radius: 4px;}
.status-row { display: flex; justify-content: space-between; align-items: center; font-size: 12px; }
.time-text { color: #a8abb2; }

/* 其他历史样式 */
.upload-area { margin: 20px 0; }
.result-box { margin-top: 20px; padding: 20px; background: #fdfdfd; border-radius: 8px; border: 1px solid #ebeef5; }
.hash-text { font-size: 12px; color: #909399; word-break: break-all; margin: 10px 0; }
.search-box { margin-bottom: 30px; }
.addr { font-family: monospace; color: #409eff; }
</style>

//npm run dev

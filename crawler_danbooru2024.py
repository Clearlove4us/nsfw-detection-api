import os
import io
import pandas as pd
import requests
from PIL import Image
from sklearn.model_selection import train_test_split
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# ==========================================
# 1. 核心工程配置
# ==========================================
PARQUET_PATH = r"D:\Users\Chenjh\Downloads\metadata.parquet"
BASE_OUTPUT_DIR = r"D:\python-learning\FalconsAI_NSFW\danbooru_dataset"
TARGET_COUNT = 10000
MIN_SCORE = 15

# ViT-base 严格要求的输入尺寸
TARGET_SIZE = 224

# 填充颜色：使用中性灰 (128, 128, 128)，这对模型归一化最友好，优于纯黑或纯白
PAD_COLOR = (128, 128, 128)
MAX_WORKERS = 32

# ==========================================
# 2. 读取、筛选并构建直链
# ==========================================
print("📂 正在读取 metadata.parquet...")
df = pd.read_parquet(PARQUET_PATH, columns=['id', 'rating', 'score', 'md5', 'file_ext'])
df = df.dropna(subset=['md5', 'file_ext'])


def build_direct_url(row):
    md5 = row['md5']
    ext = row['file_ext']
    return f"https://cdn.donmai.us/original/{md5[0:2]}/{md5[2:4]}/{md5}.{ext}"


print("🔍 正在筛选高质量图片并构建下载直链...")
df_porn = df[(df['rating'] == 'e') & (df['score'] >= MIN_SCORE)].sample(n=TARGET_COUNT, random_state=42).copy()
df_normal = df[(df['rating'] == 'g') & (df['score'] >= MIN_SCORE)].sample(n=TARGET_COUNT, random_state=42).copy()

tqdm.pandas(desc="构建URL")
df_porn['direct_url'] = df_porn.progress_apply(build_direct_url, axis=1)
df_normal['direct_url'] = df_normal.progress_apply(build_direct_url, axis=1)
del df

# ==========================================
# 3. 划分数据集 (Train: 80%, Val: 10%, Test: 10%)
# ==========================================
print("🔪 正在划分训练、验证、测试集任务...")


def prepare_split_tasks(df_source, category_name):
    train_df, temp_df = train_test_split(df_source, test_size=0.2, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    tasks = []
    for df_split, split_name in [(train_df, "train"), (val_df, "val"), (test_df, "test")]:
        dst_dir = os.path.join(BASE_OUTPUT_DIR, split_name, category_name)
        os.makedirs(dst_dir, exist_ok=True)

        # ⚠️ 关键修复：使用 img_id 接收当前行的 Index (即图片的真实 ID)
        for img_id, row in df_split.iterrows():
            tasks.append({
                'url': row['direct_url'],
                'filename': f"{img_id}.jpg",
                'dst_path': os.path.join(dst_dir, f"{img_id}.jpg")
            })
    return tasks


tasks_porn = prepare_split_tasks(df_porn, "porn")
tasks_normal = prepare_split_tasks(df_normal, "normal")
all_tasks = tasks_porn + tasks_normal


# ==========================================
# 4. 核心：Letterbox Padding 动态处理工作流
# ==========================================
def download_process_save_worker(task):
    url = task['url']
    dst_path = task['dst_path']

    if os.path.exists(dst_path):
        return

    headers = {'User-Agent': 'NSFW Classifier v0.1'}

    try:
        response = requests.get(url, headers=headers, stream=True, timeout=10)
        if response.status_code != 200:
            return

        img_buffer = io.BytesIO(response.content)

        with Image.open(img_buffer) as img:
            if img.mode == 'P':
                img = img.convert('RGB')
            if img.mode != 'RGB':
                img = img.convert('RGB')

            w, h = img.size

            # 1. 计算缩放比例 (以最长边为准)
            scale = TARGET_SIZE / max(w, h)
            new_w = int(w * scale)
            new_h = int(h * scale)

            # 2. 等比例缩小原图
            resized_img = img.resize((new_w, new_h), Image.LANCZOS)

            # 3. 创建 224x224 的中性灰背景画布
            new_img = Image.new("RGB", (TARGET_SIZE, TARGET_SIZE), PAD_COLOR)

            # 4. 计算粘贴的居中坐标并贴上原图
            paste_x = (TARGET_SIZE - new_w) // 2
            paste_y = (TARGET_SIZE - new_h) // 2
            new_img.paste(resized_img, (paste_x, paste_y))

            # 5. 保存完美处理好的张量级图片
            new_img.save(dst_path, "JPEG", quality=85, optimize=True)

    except Exception as e:
        pass


# ==========================================
# 5. 启动并发流水线
# ==========================================
print(f"🚀 开始并行全自动 Letterbox 下载处理流水线！(并发数: {MAX_WORKERS})")
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    list(tqdm(executor.map(download_process_save_worker, all_tasks), total=len(all_tasks), desc="下载并处理中"))

print("✅ 所有数据已处理完毕！")

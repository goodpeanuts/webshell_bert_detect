import os
import base64
import hashlib
import json
import pandas as pd
import logging
from sklearn.model_selection import train_test_split
from huggingface_hub import HfApi
from collections import defaultdict

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 数据集路径
white_dirs = ['../../dataset/repo/php']
black_dirs = ['../../dataset/repo/webshell', '../../dataset/manual', '../../dataset/repo/other']
# 测试范围
ext = ['php','asp', 'html', 'jsp','pl', 'aspx', 'py', 'cgi', 'sh', 'js', 'java']

# 准备数据集目录
DATASET_DIR = "webshell_dataset"
os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(f"{DATASET_DIR}/data", exist_ok=True)
os.makedirs(f"{DATASET_DIR}/samples", exist_ok=True)

def hash_file(filepath, chunk_size=8192):
    """
    返回文件内容的SHA256哈希
    :param filepath: 文件路径
    :param chunk_size: 每次读取的字节数
    :return: 文件的SHA256哈希值
    """
    sha256 = hashlib.sha256()
    try:
        with open(filepath, 'rb') as f:
            while chunk := f.read(chunk_size):
                sha256.update(chunk)
        return sha256.hexdigest()
    except (PermissionError, FileNotFoundError):
        return None

def collect_files():
    """收集良性和恶意文件的路径，并根据文件内容去重"""

    not_exist_dirs = []
    # 检查目录是否存在
    for path in white_dirs + black_dirs:
        if not os.path.exists(path):
            not_exist_dirs.append(path)
            logging.error(f"Directory {path} does not exist.")

    if not_exist_dirs:
        logging.error(f"Some directories do not exist, exit")
        exit(1)

    # 收集文件路径并去重
    def collect_and_deduplicate(dirs):
        file_hashes = defaultdict(list)
        unique_files = []
        duplicate_count = 0
        for dir in dirs:
            for root, _, files in os.walk(dir):
                for file in files:
                    if ext is None or any(file.endswith(extension) for extension in ext):
                        filepath = os.path.join(root, file)
                        file_hash = hash_file(filepath)
                        if file_hash and file_hash not in file_hashes:
                            file_hashes[file_hash].append(filepath)
                            unique_files.append(filepath)
                        elif file_hash:
                            logging.info(f"Duplicate file found: {filepath} (hash: {file_hash})")
                            duplicate_count += 1
        return unique_files, duplicate_count

    white_files, white_duplicates = collect_and_deduplicate(white_dirs)
    black_files, black_duplicates = collect_and_deduplicate(black_dirs)

    logging.info(f"↗️ Collect White files (unique): {len(white_files)}, Duplicates removed: {white_duplicates}")
    logging.info(f"↗️ Collect Black files (unique): {len(black_files)}, Duplicates removed: {black_duplicates}")
    
    return white_files, black_files, white_duplicates, black_duplicates

def label_data():
    """收集文件内容并标注为良性或恶意"""
    white_files, black_files, _, _ = collect_files()

    data = []
    file_types = {'benign': {}, 'malicious': {}}

    # 处理良性文件
    for file_path in white_files:
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                # 统计文件类型
                ext = os.path.splitext(file_path)[1].lower().lstrip('.')
                if ext in file_types['benign']:
                    file_types['benign'][ext] += 1
                else:
                    file_types['benign'][ext] = 1
                
                data.append({
                    'code': content, 
                    'code_b64': base64.b64encode(content.encode('utf-8')).decode('utf-8'),
                    'file_type': ext,
                    'file_size': len(content),
                    'label': 0,  # 良性文件标签为 0
                    'source_path': file_path
                })
        except Exception as e:
            logging.error(f"Error reading {file_path}: {e}")

    # 处理恶意文件
    for file_path in black_files:
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                # 统计文件类型
                ext = os.path.splitext(file_path)[1].lower().lstrip('.')
                if ext in file_types['malicious']:
                    file_types['malicious'][ext] += 1
                else:
                    file_types['malicious'][ext] = 1
                
                data.append({
                    'code': content, 
                    'code_b64': base64.b64encode(content.encode('utf-8')).decode('utf-8'),
                    'file_type': ext,
                    'file_size': len(content),
                    'label': 1,  # 恶意文件标签为 1
                    'source_path': file_path
                })
        except Exception as e:
            logging.error(f"Error reading {file_path}: {e}")
    
    return pd.DataFrame(data), file_types

def prepare_dataset():
    """准备并保存HuggingFace格式的数据集"""
    logging.info("🔄 Preparing dataset...")
    
    # 收集和标注数据
    df, file_types = label_data()
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # 打乱数据
    
    # 去除原始路径，避免暴露敏感信息
    source_paths = df['source_path'].copy()
    df = df.drop(columns=['source_path'])
    
    # 分层划分（训练/验证/测试）
    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)
    train_df, val_df = train_test_split(train_df, test_size=0.1, stratify=train_df['label'], random_state=42)
    
    # 保存数据集文件
    train_df.to_csv(f"{DATASET_DIR}/data/train.csv", index=False)
    val_df.to_csv(f"{DATASET_DIR}/data/val.csv", index=False)
    test_df.to_csv(f"{DATASET_DIR}/data/test.csv", index=False)
    
    # 保存典型样本（每类选5个）
    benign_samples = df[df['label'] == 0].sample(5, random_state=42)
    malicious_samples = df[df['label'] == 1].sample(5, random_state=42)
    
    # 保存典型样本，用于演示
    for i, row in benign_samples.iterrows():
        filename = f"benign_sample_{i}.{row['file_type']}.txt"
        with open(f"{DATASET_DIR}/samples/{filename}", "w", encoding="utf-8") as f:
            f.write(f"# Benign Code Sample\n\n```{row['file_type']}\n{row['code']}\n```")
    
    for i, row in malicious_samples.iterrows():
        filename = f"malicious_sample_{i}.{row['file_type']}.txt"
        with open(f"{DATASET_DIR}/samples/{filename}", "w", encoding="utf-8") as f:
            f.write(f"# Malicious Code Sample (WebShell)\n\n```{row['file_type']}\n{row['code']}\n```")
    
    # 创建数据集卡片
    dataset_card = f"""---
language:
  - en
  - zh
license: mit
task_categories:
  - text-classification
task_ids:
  - binary-classification
pretty_name: WebShell Detection Dataset
tags:
  - security
  - code
  - webshell
  - cybersecurity
---

# WebShell Detection Dataset

This dataset contains code samples for training and evaluating WebShell detection models. The code samples are sourced from both legitimate web applications and known WebShells.

## Dataset Description

- **Purpose**: Research and development of security tools to detect malicious WebShells
- **Content**: Source code samples labeled as benign (0) or malicious (1)
- **File Types**: {', '.join(ext)}

## Dataset Statistics

- **Total samples**: {len(df)}
- **Benign samples**: {len(df) - df['label'].sum()}
- **Malicious samples**: {int(df['label'].sum())}
- **Training set**: {len(train_df)}
- **Validation set**: {len(val_df)}
- **Test set**: {len(test_df)}

## Data Format

Each CSV file contains the following columns:
- `code`: Original source code (use with caution)
- `code_b64`: Base64 encoded source code (recommended for use)
- `file_type`: The file extension/type
- `file_size`: The size of the file in bytes
- `label`: Class label (0: benign, 1: malicious)

## Usage Example

```python
import pandas as pd
import base64
from datasets import load_dataset

# Load using Hugging Face datasets
dataset = load_dataset("null822/webshell-sample")
train_data = dataset["train"]

# Or load directly from CSV
df = pd.read_csv('data/train.csv')

# Decode Base64 encoded code when needed
def decode_sample(encoded_text):
    return base64.b64decode(encoded_text).decode('utf-8', errors='ignore')

# Example usage
sample_code = decode_sample(df.iloc[0]['code_b64'])
```

## File Type Distribution

### Benign Files:
{json.dumps(file_types['benign'], indent=2)}

### Malicious Files (WebShells):
{json.dumps(file_types['malicious'], indent=2)}

## Ethical Use Statement

This dataset is intended solely for cybersecurity research, education, and developing defensive tools. Any use for malicious purposes is strictly prohibited.
"""
    
    with open(f"{DATASET_DIR}/README.md", "w", encoding="utf-8") as f:
        f.write(dataset_card)
    
    # 创建数据集配置文件
    dataset_config = {
        "version": "1.0.0",
        "description": "WebShell Detection Dataset",
        "citation": "",
        "homepage": "https://huggingface.co/datasets/null822/webshell-sample",
        "license": "mit",
        "features": {
            "code": {"dtype": "string", "_type": "Value"},
            "code_b64": {"dtype": "string", "_type": "Value"},
            "file_type": {"dtype": "string", "_type": "Value"},
            "file_size": {"dtype": "int64", "_type": "Value"},
            "label": {"dtype": "int64", "_type": "Value"}
        },
        "splits": {
            "train": {"name": "train", "num_bytes": train_df['code'].str.len().sum(), "num_examples": len(train_df)},
            "validation": {"name": "validation", "num_bytes": val_df['code'].str.len().sum(), "num_examples": len(val_df)},
            "test": {"name": "test", "num_bytes": test_df['code'].str.len().sum(), "num_examples": len(test_df)}
        }
    }
    
    with open(f"{DATASET_DIR}/dataset_infos.json", "w") as f:
        json.dump({"default": dataset_config}, f, indent=2)
    
    logging.info(f"✅ Dataset prepared in {DATASET_DIR}!")
    logging.info(f"Total samples: {len(df)}")
    logging.info(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    logging.info(f"Malicious samples: {df['label'].sum()}")
    logging.info(f"Benign samples: {len(df) - df['label'].sum()}")
    
    return df

def upload_to_huggingface(repo_id="null822/webshell-sample"):
    """上传数据集到HuggingFace"""
    logging.info(f"🚀 Uploading dataset to HuggingFace: {repo_id}")
    
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        logging.error("HF_TOKEN environment variable not set. Please set it first.")
        logging.error("You can generate a token at https://huggingface.co/settings/tokens")
        exit(1)
        
    try:
        api = HfApi(token=hf_token)
        api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
        api.upload_folder(
            folder_path=DATASET_DIR,
            repo_id=repo_id,
            repo_type="dataset",
        )
        logging.info(f"✅ Dataset uploaded successfully to https://huggingface.co/datasets/{repo_id}")
    except Exception as e:
        logging.error(f"Error uploading to HuggingFace: {e}")

if __name__ == "__main__":
    prepare_dataset()
    
    # 询问是否上传
    upload = input("Do you want to upload the dataset to HuggingFace now? (y/n): ").strip().lower()
    if upload == 'y':
        repo_id = input("Enter your HuggingFace repo_id (default: null822/webshell-sample): ").strip()
        if not repo_id:
            repo_id = "null822/webshell-sample"
        upload_to_huggingface(repo_id)
    else:
        logging.info(f"Dataset prepared in {DATASET_DIR}. You can upload it manually later.")
        logging.info("Use this command to upload:")
        logging.info("python -c \"from huggingface_hub import HfApi; api = HfApi(token=os.getenv('HF_TOKEN')); api.upload_folder(folder_path='webshell_dataset', repo_id='null822/webshell-sample', repo_type='dataset')\"")

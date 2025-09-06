# up.py - Hugging Face Model Upload Script
# 
# This script automatically uploads all model folders from the ../models directory
# to a single Hugging Face repository, placing each model in its own subfolder.
# It handles retries, connection issues, and provides detailed logs of the upload process.
# 
# Usage:
#   1. Set your Hugging Face token: export HF_TOKEN="your_token"
#   2. Run the script: python up.py
#
# All models will be uploaded to the null822/webshell-sample repository.

import os
import time
import logging
from pathlib import Path
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import HfHubHTTPError
import requests.exceptions

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

def upload_model_to_hub(
    model_path="../models/", 
    repo_id="null822/webshell-detect-bert", 
    max_retries=5, 
    retry_delay=30,
    create_if_not_exists=True,
    subfolder=None
):
    """上传模型到Hugging Face，具有重试机制
    
    Args:
        model_path: 模型文件夹路径
        repo_id: Hugging Face仓库ID
        max_retries: 最大重试次数
        retry_delay: 重试延迟(秒)
        create_if_not_exists: 如果仓库不存在，是否创建
        subfolder: 上传到仓库的子文件夹路径
    """
    
    token = os.getenv("HF_TOKEN")
    if not token:
        raise ValueError("环境变量 HF_TOKEN 未设置。请设置你的Hugging Face访问令牌。")
    
    api = HfApi(token=token)
    
    # 检查仓库是否存在，如果不存在则创建
    if create_if_not_exists:
        try:
            logging.info(f"检查仓库 {repo_id} 是否存在...")
            api.repo_info(repo_id=repo_id, repo_type="model")
            logging.info(f"仓库 {repo_id} 已存在")
        except Exception:
            logging.info(f"仓库 {repo_id} 不存在，正在创建...")
            create_repo(repo_id=repo_id, repo_type="model", token=token)
            logging.info(f"仓库 {repo_id} 创建成功")
    
    # 检查模型路径是否存在
    model_path = Path(model_path)
    if not model_path.exists():
        raise ValueError(f"模型路径 {model_path} 不存在")
    
    # 列出文件大小
    total_size = 0
    file_count = 0
    for file in model_path.rglob("*"):
        if file.is_file():
            size_mb = file.stat().st_size / (1024 * 1024)
            logging.info(f"文件: {file.relative_to(model_path)}, 大小: {size_mb:.2f} MB")
            total_size += size_mb
            file_count += 1
    
    logging.info(f"总共 {file_count} 个文件，总大小: {total_size:.2f} MB")
    
    # 带重试的上传
    attempt = 0
    while attempt < max_retries:
        attempt += 1
        try:
            logging.info(f"尝试上传 ({attempt}/{max_retries})...")
            
            api.upload_folder(
                folder_path=str(model_path),
                repo_id=repo_id,
                repo_type="model",
                # 增加超时时间
                allow_patterns=["*"],
                delete_patterns=[],
                path_in_repo=subfolder,  # 上传到仓库的子文件夹
                commit_message=f"Upload model from {model_path}" + (f" to {subfolder}" if subfolder else "")
            )
            
            logging.info(f"✅ 上传成功！模型现在可在 https://huggingface.co/{repo_id} 访问")
            return True
            
        except (requests.exceptions.SSLError, requests.exceptions.ConnectionError, 
                RuntimeError, HfHubHTTPError) as e:
            if attempt < max_retries:
                logging.warning(f"上传失败 ({attempt}/{max_retries}): {str(e)}")
                logging.info(f"将在 {retry_delay} 秒后重试...")
                time.sleep(retry_delay)
            else:
                logging.error(f"已达最大重试次数 ({max_retries})，上传失败: {str(e)}")
                return False

def scan_model_folders(base_path="../models"):
    """扫描并返回所有模型文件夹的路径"""
    model_paths = []
    base_path = Path(base_path)
    
    # 检查基础路径是否存在
    if not base_path.exists():
        # 尝试一个备用路径，相对于脚本位置
        script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
        alternate_path = script_dir.parent / "models"
        if alternate_path.exists():
            base_path = alternate_path
            logging.info(f"使用备用模型路径: {base_path}")
        else:
            logging.error(f"模型基础路径不存在: {base_path} 或 {alternate_path}")
            return model_paths
    
    logging.info(f"扫描模型文件夹: {base_path}")
    
    # 检查直接子目录
    for item in base_path.iterdir():
        if item.is_dir():
            # 检查直接子目录是否是模型目录
            model_files = list(item.glob("*.bin")) + list(item.glob("*.safetensors")) + list(item.glob("config.json"))
            if model_files:
                logging.info(f"找到模型目录: {item}")
                model_paths.append(str(item))
            else:
                # 检查更深一级的目录
                for subitem in item.iterdir():
                    if subitem.is_dir():
                        model_files = list(subitem.glob("*.bin")) + list(subitem.glob("*.safetensors")) + list(subitem.glob("config.json"))
                        if model_files:
                            logging.info(f"找到模型子目录: {subitem}")
                            model_paths.append(str(subitem))
    
    return model_paths

if __name__ == "__main__":
    # 设置默认仓库
    target_repo = "null822/webshell-detect-bert"
    
    # 自动扫描模型文件夹
    models_base_path = "../models"
    model_paths = scan_model_folders(models_base_path)
    
    if not model_paths:
        logging.error(f"未找到任何模型文件夹，请检查路径: {models_base_path}")
        exit(1)
    
    # 显示找到的模型文件夹
    logging.info(f"找到 {len(model_paths)} 个模型文件夹:")
    for i, path in enumerate(model_paths, 1):
        logging.info(f"{i}. {path}")
    
    # 自动上传所有模型到同一个仓库的不同子文件夹
    logging.info(f"开始上传所有模型到仓库: {target_repo}")
    
    success_count = 0
    for path in model_paths:
        # 提取子文件夹名称
        subfolder_name = os.path.basename(path)
        # 如果是full/codebert_model这种格式，取前面的部分作为前缀
        parent_dir = os.path.basename(os.path.dirname(path))
        if parent_dir not in ["models"]:
            subfolder_name = f"{parent_dir}_{subfolder_name}"
        
        logging.info(f"开始上传: {path} -> {target_repo}/{subfolder_name}")
        success = upload_model_to_hub(
            model_path=path, 
            repo_id=target_repo, 
            subfolder=subfolder_name
        )
        
        if success:
            logging.info(f"模型 {path} 上传成功到 {target_repo}/{subfolder_name}")
            success_count += 1
        else:
            logging.error(f"模型 {path} 上传失败")
    
    # 显示上传总结
    if success_count == len(model_paths):
        logging.info(f"所有 {len(model_paths)} 个模型都成功上传到仓库 {target_repo}")
    else:
        logging.warning(f"共 {len(model_paths)} 个模型，{success_count} 个上传成功，{len(model_paths) - success_count} 个上传失败")

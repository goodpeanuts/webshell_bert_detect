import os
import logging
import pandas as pd
import hashlib
from collections import defaultdict

# 数据集路径
white_dirs = ['../../dataset/repo/php']
black_dirs = ['../../dataset/repo/webshell', '../../dataset/manual', '../../dataset/repo/other']
# 测试范围
ext = ['php','asp', 'html', 'jsp','pl', 'aspx', 'py', 'cgi', 'sh', 'js', 'java']

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

def collect_files() -> tuple[list, list, int, int]:
    """
    收集良性和恶意文件的路径，并根据文件内容去重
    :return: 良性文件路径列表、恶意文件路径列表、白文件去重删除数、黑文件去重删除数
    """

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
    """
    收集文件内容并标注为良性或恶意
    :return: 包含代码和标签的DataFrame
    """
    white_files, black_files, _, _ = collect_files()

    data = []

    for file_path in white_files:
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                data.append({'code': content, 'label': 0})  # 良性文件标签为 0
        except Exception as e:
            logging.error(f"Error reading {file_path}: {e}")

    for file_path in black_files:
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                data.append({'code': content, 'label': 1})  # 恶意文件标签为 1
        except Exception as e:
            logging.error(f"Error reading {file_path}: {e}")

    return pd.DataFrame(data)

if __name__ == "__main__":
    white_files, black_files, white_duplicates, black_duplicates = collect_files()
    print(f"白文件去重后保留: {len(white_files)}，删除重复文件数: {white_duplicates}")
    print(f"黑文件去重后保留: {len(black_files)}，删除重复文件数: {black_duplicates}")



import os
import logging
import pandas as pd

# 数据集路径
dataset_path = '../../dataset/repo/'
white = ['php']
black = ['webshell']
# 测试范围
ext = ['php']

def collect_files() -> tuple[list, list]:
    """
    收集良性和恶意文件的路径
    :return: 良性和恶意文件路径列表
    """
    white_dirs = [os.path.join(dataset_path, item) for item in white]
    black_dirs = [os.path.join(dataset_path, item) for item in black]

    not_exist_dirs = []
    # 检查目录是否存在
    for path in white_dirs + black_dirs:
        if not os.path.exists(path):
            not_exist_dirs.append(path)
            logging.error(f"Directory {path} does not exist.")

    if not_exist_dirs:
        logging.error(f"Some directories do not exist, exit")
        exit(1)


    # 收集白名单文件
    white_files = []
    for dir in white_dirs:
        for root, _, files in os.walk(dir):
            for file in files:
                if ext is None or any(file.endswith(extension) for extension in ext):
                    white_files.append(os.path.join(root, file))
    
    # 收集黑名单文件
    black_files = []
    for dir in black_dirs:
        for root, _, files in os.walk(dir):
            for file in files:
                if ext is None or any(file.endswith(extension) for extension in ext):
                    black_files.append(os.path.join(root, file))

    logging.info(f"↗️ Collect White files: {len(white_files)}")
    logging.info(f"↗️ Collect Black files: {len(black_files)}")
    
    return white_files, black_files

def label_data():

    white_files, black_files = collect_files()

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


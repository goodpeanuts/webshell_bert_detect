import os
import shutil

"""
主要功能是从源目录 (src_dir) 中复制文件到目标目录 (dest_dir)，只保留指定后缀的文件，并清理掉不符合条件的文件和空文件夹。
"""

src_directory = "./repo"
dest_directory = "./filter"
allowed_file_extensions = [".txt", ".md"]

def filter_and_clean_files(src_dir: str, dest_dir: str, allowed_extensions: list[str]):
    """
    复制 src_dir 下的所有文件到 dest_dir，保留指定后缀的文件，删除其他文件，并清理空文件夹。

    :param src_dir: 源目录路径
    :param dest_dir: 目标目录路径
    :param allowed_extensions: 允许保留的文件后缀列表（如 ['.py', '.txt']）
    """
    allowed_extensions.sort()
    dest_dir = os.path.join(dest_dir, "_".join(allowed_extensions).replace(".", ""))

    if os.path.exists(dest_dir):
        # 删除整个目录及其内容
        shutil.rmtree(dest_dir)
        # 重新创建空目录
    
    os.makedirs(dest_dir, exist_ok=True)

    # 遍历源目录下的所有文件夹
    for item in os.listdir(src_dir):
        src_path = os.path.join(src_dir, item)
        dest_path = os.path.join(dest_dir, item)

        # 如果是文件夹，则复制
        if os.path.isdir(src_path):
            shutil.copytree(src_path, dest_path, dirs_exist_ok=True)

    # 遍历目标目录，删除不符合后缀的文件
    for root, _, files in os.walk(dest_dir):
        for file in files:
            file_path = os.path.join(root, file)
            if not any(file.endswith(ext) for ext in allowed_extensions):
                os.remove(file_path)

    # 清理空文件夹
    for root, dirs, _ in os.walk(dest_dir, topdown=False):
        for dir in dirs:
            dir_path = os.path.join(root, dir)
            if not os.listdir(dir_path):  # 如果文件夹为空
                os.rmdir(dir_path)

filter_and_clean_files(src_directory, dest_directory, allowed_file_extensions)
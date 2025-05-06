import os
import hashlib
from collections import defaultdict

def hash_file(filepath, chunk_size=8192):
    """返回文件内容的SHA256哈希"""
    sha256 = hashlib.sha256()
    try:
        with open(filepath, 'rb') as f:
            while chunk := f.read(chunk_size):
                sha256.update(chunk)
        return sha256.hexdigest()
    except (PermissionError, FileNotFoundError):
        return None

def find_duplicate_files(root_dir):
    """遍历目录，找出内容相同的文件"""
    hash_to_paths = defaultdict(list)

    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            file_hash = hash_file(filepath)
            if file_hash:
                hash_to_paths[file_hash].append(filepath)

    # 只返回有重复的文件组
    duplicates = {hash: paths for hash, paths in hash_to_paths.items() if len(paths) > 1}
    return duplicates

if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("用法：python check_duplicates.py <目录路径>")
        sys.exit(1)

    target_dir = sys.argv[1]
    duplicates = find_duplicate_files(target_dir)

    if duplicates:
        print("发现重复文件：")
        for hash_value, paths in duplicates.items():
            print(f"\n哈希: {hash_value}")
            for path in paths:
                print(f" - {path}")
    else:
        print("未发现重复文件。")

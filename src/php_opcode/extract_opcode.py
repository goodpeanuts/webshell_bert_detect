from gettext import find
import subprocess
import os
import re
import csv
from typing import List
import glob
from datetime import datetime
import logging
from tqdm import tqdm
import collect
import shutil

current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
os.makedirs("./logs", exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s", filename=f"./logs/extract-opcode-{current_time}.log", filemode="w")

"""
从 PHP 扩展的 stderr.txt 文件中提取操作码表，并将其转换为 CSV 格式
"""
def parse_opcode_table(lines: List[str]) -> List[dict]:
    entries = []

    pattern = re.compile(
        r'^\s*(\d+)?\s+(\d+)\s+([ >]*)\s*(\w+)'              # line, op_index, flags, opcode
        r'(?:\s+(\w+))?'                                      # fetch type
        r'(?:\s+(\~?\d+|\w+))?'                               # ext or return
        r'(?:\s+(\~?\d+|\w+))?'                               # return or extra
        r'\s*(.*)?$'                                          # operands
    )

    for line in lines:
        if not line.strip() or re.match(r'-{5,}', line):  # skip separators
            continue

        match = pattern.match(line)
        if not match:
            continue

        line_num, op_index, flags, opcode, fetch, ext, ret, operands = match.groups()
        entries.append({
            "line": line_num,
            "op_index": op_index,
            "flag": flags.strip(),
            "op": opcode,
            "fetch": fetch,
            "ext": ext,
            "ret": ret,
            "operands": operands.strip() if operands else ""
        })

    return entries

"""
将操作码表按表格分割
"""
def split_by_opcode_tables(content: str) -> List[List[str]]:
    tables = []
    current = []
    inside = False
    for line in content.splitlines():
        if "compiled vars" in line:
            inside = True
            current = []
        elif inside and re.match(r'\s*\d+\s+\d+\s+[ >]*\w+', line):
            current.append(line)
        elif inside and current and not line.strip():  # empty line indicates table end
            tables.append(current)
            inside = False
    if current:
        tables.append(current)
    return tables


"""
从内容中提取操作码表并返回 CSV 格式的字符串
"""
def extract_opcode(content: str) -> str:
    tables = split_by_opcode_tables(content)
    output = []

    for idx, table_lines in enumerate(tables):
        rows = parse_opcode_table(table_lines)
        csv_output = []
        writer = csv.DictWriter(csv_output, fieldnames=[
            "line", "op_index", "flag", "op", "fetch", "ext", "ret", "operands"
        ])
        writer.writeheader()
        writer.writerows(rows)
        output.append("\n".join(csv_output))

    return "\n\n".join(output)

"""
从文件中提取操作码表并提取opcode
"""
def extract_from_file(filepath: str):
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    csv_content = extract_opcode(content)

    return csv_content

"""
从文件中提取操作码表并保存为 CSV 文件
"""
def extract_opcode_from_file_and_save(filepath: str, output_dir: str = "./csv_output"):
    csv_content = extract_from_file(filepath)

    # Generate filename prefix by replacing '.' and '/' with '_'
    filename_prefix = re.sub(r'[./]', '_', os.path.relpath(filepath, start=os.getcwd())).lstrip('_')

    os.makedirs(output_dir, exist_ok=True)

    tables = split_by_opcode_tables(csv_content)
    for idx, table_lines in enumerate(tables):
        rows = parse_opcode_table(table_lines)
        outpath = os.path.join(output_dir, f"{filename_prefix}_{idx}.csv")

        with open(outpath, "w", newline='', encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=[
                "line", "op_index", "flag", "op", "fetch", "ext", "ret", "operands"
            ])
            writer.writeheader()
            writer.writerows(rows)

        print(f"[+] Wrote {outpath} ({len(rows)} rows)")

    print(f"[+] Completed writing CSV files for {filepath}")

def start_docker_container() -> str | None:
    # 检查是否有已存在的容器
    existing_container_id = subprocess.check_output(
        [
            'docker', 'ps', '-q', '--filter', 'name=php_vld_container'
        ],
        text=True
    ).strip()

    if len(existing_container_id) != 0:
        logging.info(f"[+] Existing Docker container found: {existing_container_id}")
        return existing_container_id
    
    """启动一个长期运行的 Docker 容器"""
    try:
        container_id = subprocess.check_output(
            [
                'docker', 'run', '-d', '--rm',
                '-v', f"{os.getcwd()}:/app",
                '--name', 'php_vld_container',
                'php-vld', 'tail', '-f', '/dev/null'
            ],
            text=True
        ).strip()
        logging.info(f"[+] Docker container started: {container_id}")
        return container_id
    except Exception as e:
        logging.error(f"[!] Error starting Docker container: {e}")
        return None


def stop_docker_container(container_id: str):
    """停止 Docker 容器"""
    try:
        subprocess.run(['docker', 'stop', container_id], check=True)
        logging.info(f"[+] Docker container stopped: {container_id}")
    except Exception as e:
        logging.error(f"[!] Error stopping Docker container: {e}")


def container_extract_opcodes(container_id: str, file_path: str):
    """在已有的 Docker 容器中提取 PHP 文件的操作码"""
    try:
        abs_path = os.path.abspath(file_path)
        rel_path = os.path.relpath(abs_path, os.getcwd())

        # 确保文件在挂载目录内
        assert abs_path.startswith(os.getcwd()), f"File {file_path} is outside the mounted directory."

        docker_cmd = [
            'docker', 'exec', container_id,
            'php', '-d', 'vld.active=1', '-d', 'vld.execute=0',
            f"/app/{rel_path}"
        ]

        result = subprocess.run(
            docker_cmd, capture_output=True, text=True, timeout=20
        )

        stdout = result.stdout
        stderr = result.stderr

        return (stdout, stderr)

    except Exception as e:
        print(f"[!] Error extracting opcode: {e}")
        return (None, None)

def find_stderr_txt_files(directory):
    # 使用递归模式匹配 .stderr.txt 文件
    files = glob.glob(f"{directory}/**/*.stderr.txt", recursive=True)
    # 确保路径完整性
    return [os.path.normpath(file) for file in files]

def print_files():
    directory = "./filter/php"
    files = find_stderr_txt_files(directory)
    for file in files:
        assert os.path.exists(file), f"File not found: {file}"
        print(file)


"""
提取 PHP 扩展的操作码表, 分别将stdout和stderr输出到文件
"""
if __name__ == "__main__":
    container_id = start_docker_container()
    if not container_id:
        logging.error("[-] Failed to start Docker container.")
        exit(1)


    w, b, _, _ = collect.collect_files()

    if os.path.exists("repo"):
        shutil.rmtree("repo")
    os.makedirs("repo/0", exist_ok=True)
    os.makedirs("repo/1", exist_ok=True)
    new_w = []
    new_b = []

    for file_path in w:
        filename_prefix = re.sub(r'[./]', '_', os.path.relpath(file_path, start=os.getcwd())).lstrip('_')
        dest_path = os.path.join("repo/0", filename_prefix)
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        shutil.copy(file_path, dest_path)
        new_w.append(dest_path)

    for file_path in b:
        filename_prefix = re.sub(r'[./]', '_', os.path.relpath(file_path, start=os.getcwd())).lstrip('_')
        dest_path = os.path.join("repo/1", filename_prefix)
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        shutil.copy(file_path, dest_path)
        new_b.append(dest_path)

    try:
        # 多次提取操作码
        for php_file in tqdm(new_w + new_b, desc="Extracting white opcodes"):
            (stdout, stderr) = container_extract_opcodes(container_id, php_file)
            if stdout:
                out_file = os.path.splitext(php_file)[0] + "_opcode.stdout.txt"
                with open(out_file, "w") as f:
                    f.write(stdout)                
            else:
                logging.error(f"[-] Failed to extract opcode stdout for {php_file}.")

            if stderr:
                err_file = os.path.splitext(php_file)[0] + "_opcode.stderr.txt"
                with open(err_file, "w") as f:
                    f.write(stderr)
            else:
                logging.error(f"[-] Failed to extract opcode stderr for {php_file}.")
    finally:
        # 确保程序结束时停止 Docker 容器
        stop_docker_container(container_id)

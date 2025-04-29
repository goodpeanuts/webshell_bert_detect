import subprocess
import os
import filter
from datetime import datetime
import logging
from tqdm import tqdm

current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
os.makedirs("./logs", exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s", filename=f"./logs/extract-opcode-{current_time}.log", filemode="w")

def start_docker_container() -> str | None:
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


def extract_opcodes(container_id: str, file_path: str):
    """在已有的 Docker 容器中提取 PHP 文件的操作码"""
    try:
        abs_path = os.path.abspath(file_path)
        rel_path = os.path.relpath(abs_path, os.getcwd())

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

        # if 'compiled vars' in stdout:
            # lines = stdout.splitlines()
            # start = next(i for i, line in enumerate(lines) if 'compiled vars' in line)
            # return '\n'.join(lines[start:])
        return (stdout, stderr)
        # else:
        #     logging.error(f"[-] No compiled vars found in {file_path} stdout")
        #     return (None, None)

    except Exception as e:
        print(f"[!] Error extracting opcode: {e}")
        return (None, None)


# 示例用法
if __name__ == "__main__":
    container_id = start_docker_container()
    if not container_id:
        logging.error("[-] Failed to start Docker container.")
        exit(1)

    # src_directory = "./repo"
    # dest_directory = "./filter"
    src_directory = "./manual"
    dest_directory = "./filter/manual"
    allowed_file_extensions = [".php"]

    phps = filter.filter_and_clean_files(src_directory, dest_directory, allowed_file_extensions)

    try:
        # 多次提取操作码
        for php_file in tqdm(phps, desc="Extracting opcodes"):
            (stdout, stderr) = extract_opcodes(container_id, php_file)
            if stdout:
                out_file = os.path.splitext(php_file)[0] + "_opcode.stdout.txt"
                with open(out_file, "w") as f:
                    f.write(stdout)
                logging.info(f"[+] Opcode written to {out_file}")
                
            else:
                logging.error(f"[-] Failed to extract opcode stdout for {php_file}.")

            if stderr:
                err_file = os.path.splitext(php_file)[0] + "_opcode.stderr.txt"
                with open(err_file, "w") as f:
                    f.write(stderr)
                logging.info(f"[+] Opcode error written to {err_file}")
            else:
                logging.error(f"[-] Failed to extract opcode stderr for {php_file}.")
    finally:
        # 确保程序结束时停止 Docker 容器
        stop_docker_container(container_id)

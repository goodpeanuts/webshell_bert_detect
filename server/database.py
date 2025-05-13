import subprocess
import logging
import pymysql
import os

def get_db_conn():
    return pymysql.connect(
        host="127.0.0.1",
        user="wbsuser",
        port=3359,
        password="wbspass",
        database="wbs",
        charset="utf8mb4"
    )

def ensure_mysql_container():
    try:
        # 检查容器是否已启动
        result = subprocess.run(
            ["docker", "ps", "-q", "-f", "name=wbs_mysql"],
            capture_output=True, text=True
        )
        if not result.stdout.strip():
            logging.info("MySQL 容器启动中")
            # 没有启动则启动
            subprocess.run(
                ["docker", "compose", "up", "-d", "mysql"],
                cwd=os.path.dirname(os.path.abspath(__file__)),
                check=True
            )
            logging.info("MySQL 容器已启动")
        else:
            logging.info("MySQL 容器已在运行")
    except Exception as e:
        logging.error(f"检查或启动 MySQL 容器失败: {e}")

def stop_mysql_container():
    try:
        subprocess.run(
            ["docker", "compose", "stop", "mysql"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            check=False
        )
        logging.info("MySQL 容器已停止")
    except Exception as e:
        logging.error(f"MySQL 容器停止失败: {e}")
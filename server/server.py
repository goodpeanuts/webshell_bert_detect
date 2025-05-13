import logging
import atexit
import signal
import sys
from flask import Flask

from . import database
from . import route

def run_server(port: int = 8333):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )

    app = Flask(__name__)
    app.secret_key = "your_secret_key"
    app.register_blueprint(route.bp)

    database.ensure_mysql_container()
    # 注册退出时自动关闭数据库容器
    atexit.register(database.stop_mysql_container)

    def handle_exit(signum, frame):
        logging.info("Received exit signal, shutting down...")
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)

    app.run(host="0.0.0.0", port=port)


if __name__ == "__main__":
    run_server()
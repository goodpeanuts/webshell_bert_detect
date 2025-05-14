import json
import logging
from gradio_client import Client
import uuid
import os
from datetime import datetime
from functools import wraps
from flask import Blueprint, request, jsonify, render_template, abort, session, redirect, url_for

from . import database

bp = Blueprint("main", __name__)

def get_gradio_client():
    if not hasattr(get_gradio_client, "_client"):
        get_gradio_client._client = Client("http://127.0.0.1:7860/")
    return get_gradio_client._client

# 检查并创建shells文件夹
SHELLS_DIR = "shells"
os.makedirs(SHELLS_DIR, exist_ok=True)

@bp.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        conn = database.get_db_conn()
        with conn.cursor() as cursor:
            cursor.execute("SELECT password FROM users WHERE username=%s", (username,))
            row = cursor.fetchone()
        conn.close()
        if row and row[0] == password:
            session["user"] = username
            return redirect(url_for("main.list_shells"))
        else:
            return render_template("login.html", error="用户名或密码错误")
    return render_template("login.html")

@bp.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("main.login"))

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if "user" not in session:
            return redirect(url_for("main.login"))
        return f(*args, **kwargs)
    return decorated_function

@bp.route("/")
@login_required
def list_shells():
    filter_ip = request.args.get("ip", "").strip()
    records = []
    for fname in os.listdir(SHELLS_DIR):
        fpath = os.path.join(SHELLS_DIR, fname)
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
                records.append({
                    "ip": data.get("ip", ""),
                    "path": data.get("path", ""),
                    "time": data.get("time", "")
                })
        except Exception as e:
            logging.warning(f"Failed to load {fpath}: {e}")

    records.sort(key=lambda x: x["time"], reverse=True)
    if filter_ip:
        records = [r for r in records if filter_ip in r["ip"]]

    return render_template("warning_list.html", items=records, filter_ip=filter_ip)

@bp.route("/detail")
@login_required
def detail():
    path = request.args.get("path", "")
    # 在shells目录查找包含该path的文件
    for fname in os.listdir(SHELLS_DIR):
        fpath = os.path.join(SHELLS_DIR, fname)
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
                if data.get("path") == path:
                    return render_template("detail.html", item=data)
        except Exception as e:
            continue
    return abort(404, description="File not found")

@bp.route("/predict", methods=["POST"])
# @login_required
def predict():
    try:
        req = request.get_json(force=True)
        req_id = req["id"]
        path = req["path"]
        content = req["content"]
        client_ip = request.remote_addr

        logging.info(f"Received request ID: {req_id}, Path: {path}, From: {client_ip}")

        # 调用 gradio client 预测
        gradio_client = get_gradio_client()
        result = gradio_client.predict(code=content, api_name="/predict")
        logging.info(f"Prediction result: {result}")

        # 如果是恶意代码则保存到shells文件夹
        if result == "恶意 WebShell":
            file_uuid = str(uuid.uuid4())
            save_path = os.path.join(SHELLS_DIR, file_uuid)
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            file_data = {
                "ip": client_ip,
                "time": now,
                "path": path,
                "content": content
            }
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(file_data, f, ensure_ascii=False, indent=2)
            logging.info(f"Malicious code saved to: {save_path}")

        resp = {
            "id": req_id,
            "result": result
        }
        return jsonify(resp)
    except Exception as e:
        logging.error(f"Error handling request: {e}")
        return jsonify({"id": "", "result": f"error: {str(e)}"}), 500
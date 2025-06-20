from pdb import run
from numpy import block
import torch
import transformers
import gradio as gr
import multiprocessing
import sys
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from transformers import AutoModel, AutoTokenizer
from fvcore.nn import FlopCountAnalysis
from server.server import run_server

model_path = 'src/php_tiny/tinybert_model'


def show_model_info(path: str):

    model = AutoModel.from_pretrained(path)
    name = model.__class__.__name__
    print(f"Model name: {name}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"总参数量: {total_params / 1e6:.2f}M")

    batch_size = 1
    seq_length = 512
    dummy_input_ids = torch.ones(batch_size, seq_length, dtype=torch.long)

    flops = FlopCountAnalysis(model, (dummy_input_ids,))

    print(f"FLOPs: {flops.total() / 1e9:.2f} GFLOPs")


def run_web(path: str, port:  int = 7860):
    # 加载模型
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path)
    model.eval()

    def classify_code(code: str):
        inputs = tokenizer(code, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
            pred = torch.argmax(outputs.logits, dim=1).item()
        return "恶意 WebShell" if pred == 1 else "正常代码"

    def classify_file(file):
        with open(file, "r", encoding="utf-8", errors="ignore") as f:
            code = f.read()
        return classify_code(code)

    # Gradio 界面
    iface = gr.Interface(
        fn=classify_code,
        inputs=gr.Textbox(lines=10, label="输入代码"),
        outputs=gr.Textbox(label="检测结果"),
        title="WebShell 检测器（文本输入）"
    )

    file_iface = gr.Interface(
        fn=classify_file,
        inputs=gr.File(label="上传 PHP 文件"),
        outputs=gr.Textbox(label="检测结果"),
        title="WebShell 检测器（文件上传）"
    )

    # 合并两个界面
    gr.TabbedInterface([iface, file_iface], ["文本检测", "文件检测"]).launch(
        server_name="0.0.0.0", server_port=port
    )

if __name__ == "__main__":
    print(f"cuda.is_available(): {torch.cuda.is_available()}")
    print(f"transformers 版本 {transformers.__version__}")
    show_model_info(model_path)

    # 启动 Flask 服务到新进程
    server_proc = multiprocessing.Process(target=run_server, kwargs={"port": 8333})
    server_proc.start()

    try:
        run_web(model_path)  # block=True，主进程阻塞
    except KeyboardInterrupt:
        print("收到退出信号...")
    finally:
        sys.exit(0)

import os
import logging
import torch
from datetime import datetime
from transformers import RobertaTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import collect

current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = "./logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s", filename=f"./logs/test-{current_time}.log", filemode="w")

model_path = "codebert_model"
tokenizer = RobertaTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type != "cuda":
    logging.info("❌ CUDA is not available. Exiting...")
    exit(1)

model.to(device)
model.eval()

def eval_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # 计算准确率（Accuracy）
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    # 计算精确率（Precision）
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0

    # 计算召回率（Recall）
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    # 计算 F1 分数
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    logging.info(f"Accuracy: {accuracy:.4f}")
    logging.info(f"Precision: {precision:.4f}")
    logging.info(f"Recall: {recall:.4f}")
    logging.info(f"F1 Score: {f1:.4f}")



# 预测函数
def predict_php_file(filepath):
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        code = f.read()

    # 编码
    inputs = tokenizer(code, return_tensors="pt", padding="max_length", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # 预测
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()
        confidence = torch.softmax(logits, dim=1)[0][predicted_class].item()

    return predicted_class, confidence

def test_php_file(filepath):
    # 检查文件是否存在
    if not os.path.isfile(filepath):
        logging.error(f"❌ File not found: {filepath}")
        return -1, -1

    # 预测
    label, confidence = predict_php_file(filepath)
    return label, confidence

def test():
    logging.info(f"test file extension include: {collect.ext}")

    white_files, black_files, _, _ = collect.collect_files()

    black_cnt = 0
    black_correct = 0
    white_cnt = 0
    white_correct = 0
    
    err = []
    y_true = []
    y_pred = []

    start_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    logging.info(f"Test started at {start_time}")

    # 处理白名单文件
    logging.info(f"Testing normal files (0):")
    for file_path in tqdm(white_files, desc="Normal files"):
        label, confidence = test_php_file(file_path)
        white_cnt += 1
        y_true.append(0)
        y_pred.append(label)
        if label == -1:
            err.append(file_path)
            logging.error(f"File not found: {file_path}")
            continue
        elif label == 0:
            white_correct += 1
        else:
            err.append(file_path)
            logging.error(f"expect Normal: {file_path} (Predicted: {label}, Confidence: {confidence:.4f})")

    # 处理黑名单文件
    logging.info(f"Testing malicious files (1):")
    for file_path in tqdm(black_files, desc="Malicious files"):
        label, confidence = test_php_file(file_path)
        black_cnt += 1
        y_true.append(1)
        y_pred.append(label)
        if label == -1:
            err.append(file_path)
            logging.error(f"File not found: {file_path}")
            continue
        elif label == 1:
            black_correct += 1
        else:
            err.append(file_path)
            logging.error(f"expect Malicious: {file_path} (Predicted: {label}, Confidence: {confidence:.4f})")

    end_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    logging.info(f"Test ended at {end_time}")
    logging.info(f"Test duration: {datetime.now() - datetime.strptime(start_time, '%Y%m%d_%H%M%S')}")

    total_params = sum(p.numel() for p in model.parameters())
    logging.info(f"总参数量: {total_params / 1e6:.2f}M")

    batch_size = 1
    seq_length = 512
    dummy_input_ids = torch.ones(batch_size, seq_length, dtype=torch.long).to(device)

    from fvcore.nn import FlopCountAnalysis
    flops = FlopCountAnalysis(model, (dummy_input_ids,))
    logging.info(f"FLOPs: {flops.total() / 1e9:.2f} GFLOPs")
    
    correct = black_correct + white_correct
    total = black_cnt + white_cnt
    accuracy = correct / total
    logging.info(f"✅ Accuracy: {accuracy:.4f} | Total: {total} | Correct: {correct}")
    logging.info(f"Include type: {collect.ext} | Webshell: {black_cnt} | Normal: {white_cnt}")

    logging.info(f"Webshell: {black_cnt} | correct: {black_correct} | error: {black_cnt - black_correct} | Accuracy {black_correct / black_cnt:.4f}")
    logging.info(f"Normal: {white_cnt} | correct: {white_correct} | error: {white_cnt - white_correct} | Accuracy {white_correct / white_cnt:.4f}")
    for file_path in err:
        logging.error(f"Error file: {file_path}")
    
    eval_score(y_true, y_pred)


if __name__ == "__main__":
    test()

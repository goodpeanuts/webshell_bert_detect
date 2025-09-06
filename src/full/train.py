import os
from sklearn.model_selection import train_test_split
import torch
import logging
from datetime import datetime
from transformers import RobertaTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import collect

# ---------------------
# 日志设置
# ---------------------

current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
os.makedirs("./logs", exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s", filename=f"./logs/train-{current_time}.log", filemode="w")

# ---------------------
# 数据集准备
# ---------------------

import base64
from datasets import load_dataset

logging.info("📁 Loading dataset from Hugging Face...")

# 替换为你的Hugging Face数据集路径
DATASET_PATH = "null822/webshell-sample"

try:
    # 直接从Hugging Face加载数据集
    dataset = load_dataset(DATASET_PATH)
    
    # 获取各个分割
    train_raw = dataset["train"]
    val_raw = dataset["validation"] if "validation" in dataset else dataset["val"]
    test_raw = dataset["test"]
    
    logging.info(f"✅ Dataset loaded successfully from {DATASET_PATH}")
    logging.info(f"✅ Train size: {len(train_raw)} | Val: {len(val_raw)} | Test: {len(test_raw)}")

except Exception as e:
    logging.error(f"❌ Failed to load dataset from Hugging Face: {e}")
    logging.info("⚠️ Falling back to local dataset loading...")
    
    # 如果从Hugging Face加载失败，回退到本地数据集加载
    df = collect.label_data()
    df = df.sample(frac=1).reset_index(drop=True)  # shuffle
    
    # 删除空代码
    df = df.dropna(subset=["code"])
    
    # 确保 code 字段是字符串
    df["code"] = df["code"].astype(str)
    
    # 分层划分（训练/验证/测试）
    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)
    train_df, val_df = train_test_split(train_df, test_size=0.1, stratify=train_df['label'], random_state=42)
    
    # 创建Dataset对象
    train_raw = Dataset.from_pandas(train_df)
    val_raw = Dataset.from_pandas(val_df)
    test_raw = Dataset.from_pandas(test_df)
    
    logging.info(f"✅ Local dataset loaded. Train size: {len(train_raw)} | Val: {len(val_raw)} | Test: {len(test_raw)}")

# ---------------------
# 加载 tokenizer & 数据处理
# ---------------------

tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")

def tokenize_function(example):
    # 如果是从Hugging Face加载的数据集，先检查是否需要解码Base64
    if "code_b64" in example and "code" not in example:
        try:
            example["code"] = base64.b64decode(example["code_b64"]).decode("utf-8", errors="ignore")
        except:
            # 如果解码失败，使用原始的code_b64字段
            example["code"] = example["code_b64"]
    
    return tokenizer(example["code"], padding="max_length", truncation=True, max_length=512)

# 对数据集应用分词
train_ds = train_raw.map(tokenize_function, batched=True)
val_ds = val_raw.map(tokenize_function, batched=True)
test_ds = test_raw.map(tokenize_function, batched=True)

# 确保数据集中包含label列
if "label" not in train_ds.column_names and "labels" in train_ds.column_names:
    train_ds = train_ds.rename_column("labels", "label")
if "label" not in val_ds.column_names and "labels" in val_ds.column_names:
    val_ds = val_ds.rename_column("labels", "label")
if "label" not in test_ds.column_names and "labels" in test_ds.column_names:
    test_ds = test_ds.rename_column("labels", "label")

# 设置数据集格式为torch张量
train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
val_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
test_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
logging.info(f"✅ Train dataset size: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

# ---------------------
# 加载模型 & 训练参数
# ---------------------

logging.info("🧠 Loading model...")

model = AutoModelForSequenceClassification.from_pretrained("microsoft/codebert-base", num_labels=2)

training_args = TrainingArguments(
    output_dir="./results",
    logging_dir="./logs",
    logging_steps=50,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=4,
    weight_decay=0.01,
    fp16=torch.cuda.is_available(),  # 使用混合精度加速训练（如果有GPU）
    report_to="tensorboard"
)

def compute_metrics(p):
    preds = torch.argmax(torch.tensor(p.predictions), axis=1)
    labels = p.label_ids
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

logging.info(f"⚙️ Training arguments: {training_args}")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    compute_metrics=compute_metrics,
)

# ---------------------
# 训练模型
# ---------------------

logging.info("🚀 Training...")
trainer.train()

# ---------------------
# 模型评估
# ---------------------

logging.info("🧪 Evaluating on test set...")
metrics = trainer.evaluate(eval_dataset=test_ds)
logging.info(metrics)

# ---------------------
# 保存模型和 tokenizer
# ---------------------

logging.info("💾 Saving model...")
trainer.save_model("codebert_model")
tokenizer.save_pretrained("codebert_model")

logging.info("✅ Done! You can now run: tensorboard --logdir=./logs")
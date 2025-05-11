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

white = ['php']
black = ['webshell']

logging.info("📁 Loading dataset...")
df = collect.label_data()
df = df.sample(frac=1).reset_index(drop=True)  # shuffle


# 删除空代码
df = df.dropna(subset=["code"])

# 确保 code 字段是字符串
df["code"] = df["code"].astype(str)

# 分层划分（训练/验证/测试）
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)
train_df, val_df = train_test_split(train_df, test_size=0.1, stratify=train_df['label'], random_state=42)

logging.info(f"✅ Train size: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

# ---------------------
# 加载 tokenizer & 数据处理
# ---------------------

tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")

def tokenize_function(example):
    return tokenizer(example["code"], padding="max_length", truncation=True, max_length=512)

train_ds = Dataset.from_pandas(train_df).map(tokenize_function, batched=True)
val_ds = Dataset.from_pandas(val_df).map(tokenize_function, batched=True)
test_ds = Dataset.from_pandas(test_df).map(tokenize_function, batched=True)

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
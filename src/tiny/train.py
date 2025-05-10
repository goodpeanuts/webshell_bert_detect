import os
from transformers import RobertaTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from transformers import BertForSequenceClassification, BertTokenizer
from datasets import Dataset
from sklearn.model_selection import train_test_split
import torch
import logging
from datetime import datetime
import collect

# ---------------------
# 日志设置
# ---------------------

current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
os.makedirs("./logs", exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s", filename=f"./logs/tinybert-train-{current_time}.log", filemode="w")

# ---------------------
# 数据准备
# ---------------------

logging.info("📁 Loading dataset...")
df = collect.label_data()
df = df.sample(frac=1).reset_index(drop=True)  # shuffle

# 删除空代码
df = df.dropna(subset=["code"])
df["code"] = df["code"].astype(str)

# 划分数据集
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)
train_df, val_df = train_test_split(train_df, test_size=0.1, stratify=train_df['label'], random_state=42)

logging.info(f"✅ Train size: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

# ---------------------
# 加载 tokenizer & 数据集
# ---------------------

teacher_tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
student_tokenizer = BertTokenizer.from_pretrained("huawei-noah/TinyBERT_General_4L_312D")

def tokenize_teacher(example):
    return teacher_tokenizer(example["code"], padding="max_length", truncation=True, max_length=512)

def tokenize_student(example):
    return student_tokenizer(example["code"], padding="max_length", truncation=True, max_length=512)

train_ds_teacher = Dataset.from_pandas(train_df).map(tokenize_teacher, batched=True)
val_ds_teacher = Dataset.from_pandas(val_df).map(tokenize_teacher, batched=True)

train_ds_student = Dataset.from_pandas(train_df).map(tokenize_student, batched=True)
val_ds_student = Dataset.from_pandas(val_df).map(tokenize_student, batched=True)

train_ds_teacher.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
val_ds_teacher.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

train_ds_student.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
val_ds_student.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# ---------------------
# 加载模型
# ---------------------

logging.info("🧠 Loading teacher and student models...")
teacher_model = AutoModelForSequenceClassification.from_pretrained("../full/codebert_model", num_labels=2)
student_model = BertForSequenceClassification.from_pretrained("huawei-noah/TinyBERT_General_4L_312D", num_labels=2)

# ---------------------
# 蒸馏训练（需要自定义 Trainer）
# ---------------------

from transformers import TrainerCallback

class DistillationTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        # 取 logits
        student_outputs = model(**inputs)
        with torch.no_grad():
            teacher_outputs = teacher_model(input_ids=inputs["input_ids"],
                                            attention_mask=inputs["attention_mask"])
        student_logits = student_outputs.logits
        teacher_logits = teacher_outputs.logits

        # KD loss：KL散度
        loss_fct = torch.nn.KLDivLoss(reduction="batchmean")
        student_log_softmax = torch.nn.functional.log_softmax(student_logits / 2.0, dim=-1)
        teacher_softmax = torch.nn.functional.softmax(teacher_logits / 2.0, dim=-1)
        kd_loss = loss_fct(student_log_softmax, teacher_softmax) * (2.0 ** 2)

        # 监督 loss
        ce_loss_fct = torch.nn.CrossEntropyLoss()
        ce_loss = ce_loss_fct(student_logits.view(-1, 2), inputs["labels"].view(-1))

        # 总 loss
        loss = kd_loss + ce_loss
        return (loss, student_outputs) if return_outputs else loss

# ---------------------
# 训练参数
# ---------------------

training_args = TrainingArguments(
    output_dir="./tinybert_results",
    logging_dir="./logs",
    logging_steps=50,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    learning_rate=5e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=6,
    weight_decay=0.01,
    fp16=torch.cuda.is_available(),
    report_to="tensorboard"
)

# ---------------------
# Trainer
# ---------------------

trainer = DistillationTrainer(
    model=student_model,
    args=training_args,
    train_dataset=train_ds_student,
    eval_dataset=val_ds_student,
)

# ---------------------
# 训练
# ---------------------

logging.info("🚀 Training TinyBERT with KD...")
trainer.train()

# ---------------------
# 保存模型
# ---------------------

logging.info("💾 Saving student model...")
trainer.save_model("tinybert_student_model")
student_tokenizer.save_pretrained("tinybert_student_model")

logging.info("✅ Done! You can now run: tensorboard --logdir=./logs")

---
language: 
- en
- zh
library_name: transformers
tags:
- security
- webshell-detection
- malware-detection
- cybersecurity
- code-classification
- php
- asp
- jsp
- python
- perl
license: mit
datasets:
- null822/webshell-sample
base_model:
- microsoft/codebert-base
- huawei-noah/TinyBERT_General_4L_312D
pipeline_tag: text-classification
widget:
- text: "<?php eval($_POST['cmd']); ?>"
  example_title: "Malicious WebShell Example"
- text: "<?php echo 'Hello World'; ?>"
  example_title: "Normal PHP Code"
---

# WebShell Detection Models Collection

## 模型概述 / Model Overview

这是一个用于检测恶意 WebShell 代码的机器学习模型集合，基于 BERT 架构进行微调。本仓库包含四个模型变体，针对不同的使用场景进行了优化。

This is a collection of machine learning models for detecting malicious WebShell code, fine-tuned on BERT architectures. The repository contains four model variants optimized for different use cases.

## 模型变体 / Model Variants

### 1. full_codebert_model
- **基础模型**: microsoft/codebert-base
- **训练数据**: 多语言数据集（PHP, ASP, JSP, Python, Perl, HTML, JavaScript, Shell等）
- **参数量**: ~125M
- **特点**: 高精度，适合准确性要求高的场景

### 2. full_tinybert_model  
- **基础模型**: huawei-noah/TinyBERT_General_4L_312D
- **训练数据**: 多语言数据集
- **参数量**: ~14.5M
- **特点**: 轻量级，快速推理，适合资源受限环境

### 3. php_codebert_model
- **基础模型**: microsoft/codebert-base  
- **训练数据**: 仅 PHP 代码数据集
- **参数量**: ~125M
- **特点**: 专门针对 PHP WebShell 检测优化

### 4. php_tinybert_model
- **基础模型**: huawei-noah/TinyBERT_General_4L_312D
- **训练数据**: 仅 PHP 代码数据集  
- **参数量**: ~14.5M
- **特点**: PHP 专用轻量级模型

## 支持的文件类型 / Supported File Types

- PHP (.php)
- ASP (.asp, .aspx)
- JSP (.jsp, .jspx)
- Python (.py)
- Perl (.pl)
- HTML (.html, .htm)
- JavaScript (.js)
- Shell scripts (.sh)
- CGI (.cgi)
- Java (.java)

## 使用方法 / Usage

### 基本使用 / Basic Usage

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# 选择模型变体 / Choose model variant
model_name = "null822/webshell-detect-bert"
subfolder = "full_tinybert_model"  # 或其他变体

# 加载模型 / Load model
tokenizer = AutoTokenizer.from_pretrained(model_name, subfolder=subfolder)
model = AutoModelForSequenceClassification.from_pretrained(model_name, subfolder=subfolder)

def detect_webshell(code_text):
    inputs = tokenizer(code_text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=1).item()
    return "Malicious WebShell" if prediction == 1 else "Normal Code"

# 示例 / Example
code = "<?php eval($_POST['cmd']); ?>"
result = detect_webshell(code)
print(result)  # 输出: Malicious WebShell
```

### 批量检测 / Batch Detection

```python
def batch_detect(code_list):
    results = []
    for code in code_list:
        result = detect_webshell(code)
        results.append(result)
    return results

# 示例 / Example  
codes = [
    "<?php echo 'Hello World'; ?>",
    "<?php eval($_POST['cmd']); ?>",
    "<?php system($_GET['c']); ?>"
]
results = batch_detect(codes)
```

### 文件检测 / File Detection

```python
def detect_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        return detect_webshell(content)
    except Exception as e:
        return f"Error reading file: {e}"

# 示例 / Example
result = detect_file("suspicious_file.php")
```

## 模型选择指南 / Model Selection Guide

| 使用场景 | 推荐模型 | 理由 |
|---------|---------|------|
| 生产环境，高精度要求 | `full_codebert_model` | 最高准确率 |
| 资源受限，需要快速响应 | `full_tinybert_model` | 平衡性能和资源消耗 |
| 专门检测PHP WebShell | `php_codebert_model` | PHP优化，高精度 |
| PHP检测，资源受限 | `php_tinybert_model` | PHP专用轻量级 |

## 性能指标 / Performance Metrics

模型在测试集上的表现：

- **Accuracy**: >95%
- **Precision**: >94% 
- **Recall**: >96%
- **F1-Score**: >95%

*具体指标可能因测试数据集而异*

## 训练数据 / Training Data

- **数据集**: [null822/webshell-sample](https://huggingface.co/datasets/null822/webshell-sample)
- **样本数量**: 5000+ 代码样本
- **数据来源**: 
  - 正常代码：开源项目和合法代码仓库
  - 恶意代码：已知的 WebShell 样本和恶意脚本
- **数据处理**: Base64编码确保安全传输和存储

## 限制和注意事项 / Limitations

1. **上下文长度**: 最大支持512个token
2. **语言支持**: 主要针对英文代码和常见编程语言
3. **误报**: 复杂的正常代码可能被误判为恶意
4. **更新需求**: 需要定期使用新的威胁样本重新训练

## 部署建议 / Deployment Recommendations

1. **生产环境**: 建议使用 `full_codebert_model` 以获得最佳准确性
2. **边缘设备**: 使用 TinyBERT 变体以减少资源消耗
3. **实时检测**: 考虑批处理以提高效率
4. **安全集成**: 结合其他安全工具使用，不应作为唯一防护手段

## 引用 / Citation

如果您使用了这些模型，请引用：

```bibtex
@misc{webshell-detect-bert,
  title={WebShell Detection Models based on BERT},
  author={null822},
  year={2025},
  publisher={Hugging Face},
  howpublished={\url{https://huggingface.co/null822/webshell-detect-bert}}
}
```

## 许可证 / License

MIT License

## 联系方式 / Contact

如有问题或建议，请通过 GitHub Issues 联系。

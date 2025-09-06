# WebShell Detection Model

This repository contains a machine learning pipeline for detecting malicious WebShell code using fine-tuned models based on BERT architecture (CodeBERT and TinyBERT). The project includes data collection, preprocessing, model training, evaluation, and deployment with both web interface and API server capabilities.

---

## Project Overview

WebShells are malicious scripts that enable remote access and control of web servers. This project leverages deep learning models to automatically detect such malicious code across multiple programming languages, with a focus on PHP files. The detection system can be deployed as both an interactive web interface and a server accepting API requests.

### Dataset

The dataset used for training and evaluation is available on Hugging Face:
- **URL**: [null822/webshell-sample](https://huggingface.co/datasets/null822/webshell-sample)
- **Contents**: Over 5,000 code samples (both malicious and benign)
- **Format**: CSV files with Base64-encoded samples to ensure safe handling
- **Usage**: See the dataset README for detailed usage instructions

```python
# Example of loading and using the dataset
import pandas as pd
import base64

# Load dataset from Hugging Face
from datasets import load_dataset
dataset = load_dataset("your_username/webshell-detection-dataset")

# Access the data
train_data = dataset["train"]
val_data = dataset["val"]
test_data = dataset["test"]

# Decode a sample when needed
def decode_sample(encoded_text):
    return base64.b64decode(encoded_text).decode('utf-8', errors='ignore')

# Example usage
sample_code = decode_sample(train_data[0]["code_b64"])
```

### Supported Languages

The model supports detection across multiple web languages:
- PHP
- ASP/ASPX
- JSP/JSPX
- Python
- Perl
- HTML
- JavaScript
- Shell scripts
- CGI
- Java

---

### Features

- **Data Collection**: Automatically clones repositories containing benign and malicious code samples.
- **Data Preprocessing**: Filters and cleans files based on extensions and prepares labeled datasets.
- **Model Options**:
  - **CodeBERT**: Higher accuracy with larger model size
  - **TinyBERT**: Faster inference with smaller footprint
- **Evaluation**: Provides metrics such as accuracy, precision, recall, and F1 score.
- **Deployment Options**:
  - **Gradio Web Interface**: User-friendly UI for manual file analysis
  - **HTTP Server**: API endpoint for programmatic integration
  - **Shell Management**: Records and displays detected WebShells
---


## Installation

### Prerequisites

- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- CUDA (optional, for GPU acceleration)

### Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/wbs.git
   cd wbs
   ```

2. Install dependencies:
   ```bash
   uv sync
   ```

3. activate the virtual environment (if using):
   ```bash
   source .venv/bin/activate
   ```

---

## Usage

### Running the Detection System

The system can be started with a single command:

```bash
python main.py
```

This command will:
1. Load the model (default: TinyBERT)
2. Start a Gradio web interface on port 7860
3. Launch an HTTP server on port 8333 for API requests

### Using the Web Interface

The web interface is available at http://localhost:7860 and offers two ways to analyze code:

1. **Text Input**: Paste code directly into the text area
2. **File Upload**: Upload files for analysis

### Using the API Server

The HTTP server accepts POST requests to `/predict` with JSON payloads:

```bash
curl -X POST http://localhost:8333/predict \
  -H "Content-Type: application/json" \
  -d '{"id": "unique-id", "path": "/path/to/file.php", "content": "<?php echo \"Hello\"; ?>"}'
```

Response format:
```json
{
  "id": "unique-id",
  "result": "正常代码"  # or "恶意 WebShell"
}
```

### Viewing Detected WebShells (with [kspy](https://github.com/goodpeanuts/kspy))

All detected malicious files are stored in the `shells` directory with metadata. 
You can view a list of all detections by accessing:

```
http://localhost:8333/
```

This page displays IP addresses, file paths, and detection timestamps, with filtering options.

---

## Model Training

To train a new model:

### For CodeBERT

```bash
cd src/full
python train.py
```

### For TinyBERT

```bash
cd src/full_tiny
python train.py
```

---

## Deployment

For production deployment, consider:

1. Using a production WSGI server like Gunicorn
2. Setting up proper authentication for the API endpoints
3. Configuring secure HTTPS

Example production deployment:

```bash
gunicorn -w 4 -b 0.0.0.0:8333 "server.server:create_app()"
```

---

## Project Structure

```
wbs/
├── main.py                   # Main entry point for running the detection system
├── pyproject.toml            # Project dependencies and metadata
├── README.md                 # Project documentation
├── up.py                     # Model upload script for Hugging Face
│
├── dataset/                  # Data collection and preprocessing
│   ├── crawl.py              # Crawls repositories for training data
│   ├── data_php.py           # PHP dataset processing
│   ├── data_webshell.py      # WebShell dataset processing
│   ├── filter.py             # Dataset filtering utilities
│   ├── check_dup.py          # Duplicate detection
│   └── upload.py             # Dataset upload utilities
│
├── server/                   # HTTP server implementation
│   ├── server.py             # Flask server for API endpoints
│   ├── route.py              # API route definitions
│   └── templates/            # Web interface templates
│
├── shells/                   # Storage for detected malicious WebShells
│   └── [uuid files]          # Detected WebShells with metadata
│
└── src/                      # Model training and evaluation code
    ├── full/                 # CodeBERT model code
    │   ├── train.py          # Training script for CodeBERT
    │   ├── collect.py        # Data collection script
    │   ├── upload_dataset.py # Dataset upload to Hugging Face
    │   └── codebert_model/   # Saved model files
    │
    ├── php_raw/              # Raw PHP model
    │   └── test.py           # Testing script
    │
    ├── php_tiny/             # TinyBERT for PHP
    │   └── tinybert_model/   # Saved TinyBERT model files
    │
    └── full_tiny/            # TinyBERT for all languages
        └── train.py          # Training script for TinyBERT
```

### Webshell Sample Statistics

Here are some statistics from the dataset used in this project:

```bash
# dataset/repo/webshell/
find . -type f -name '*.*' | awk -F. '{print $NF}' | sort | uniq -c | sort -rn
```

```bash
   2488 php
   1503 txt
    707 jsp
    624 asp
    283 aspx
    247 rar
    204 md
    178 jpg
    129 png
     87 py
     84 pl
     75 ps1
     51 java
     40 zip
     34 cfm
     26 html
     24 jar
     23 jspx
     22 war
     21 xml
     20 cgi
     20 c
     18 class
     18 ccc
     12 vbs
     12 gitignore
     12 ashx
     11 pdf
     11 exe
     10 sh
     10 htm
      9 css
    ...
```

---

## Hugging Face Model Upload

To upload trained models to Hugging Face:

```bash
# Set your Hugging Face token as an environment variable
export HF_TOKEN="your_hugging_face_token"

# Method 1: Run the upload script directly
python up.py

# Method 2: Use the convenience shell script
./upload-to-huggingface.sh
```

The upload script automatically:
1. Scans the `models` directory for model folders
2. Uploads all model folders to a single Hugging Face repository (`null822/webshell-sample`)
3. Places each model in a separate subfolder named after the model directory
4. Handles retries and connection issues automatically
5. Provides detailed logs of the upload process

### Uploaded Models

The models are available on Hugging Face:
- **Repository**: [null822/webshell-detect-bert](https://huggingface.co/null822/webshell-detect-bert/tree/main)
- **Models**:
  - `full_codebert_model`: CodeBERT model trained on multi-language dataset
  - `full_tiny_tinybert_model`: TinyBERT model trained on multi-language dataset
  - `php_codebert_model`: CodeBERT model trained on PHP-only dataset
  - `php_tiny_tinybert_model`: TinyBERT model trained on PHP-only dataset

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.
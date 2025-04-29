# WebShell Detection Model

This repository contains a machine learning pipeline for detecting malicious WebShell code using a fine-tuned CodeBERT model. The project includes data collection, preprocessing, model training, evaluation, and deployment.

---

## Features

- **Data Collection**: Automatically clones repositories containing benign and malicious PHP code.
- **Data Preprocessing**: Filters and cleans files based on extensions and prepares labeled datasets.
- **Model Training**: Fine-tunes the CodeBERT model for binary classification (malicious vs. benign).
- **Evaluation**: Provides metrics such as accuracy, precision, recall, and F1 score.
- **Deployment**: Includes a Gradio-based web interface for real-time WebShell detection.

---

## Project Structure

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
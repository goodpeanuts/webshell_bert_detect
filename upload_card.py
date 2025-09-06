from huggingface_hub import HfApi

api = HfApi()

# 读取模型卡片内容
with open('/home/dell/repo/wbs/MODEL_CARD.md', 'r', encoding='utf-8') as f:
    readme_content = f.read()

# 上传到仓库
api.upload_file(
    path_or_fileobj=readme_content.encode(),
    path_in_repo="README.md",
    repo_id="null822/webshell-detect-bert",
    repo_type="model",
    commit_message="Add comprehensive model card"
)
import torch
import transformers

from transformers import AutoModel, AutoTokenizer
from fvcore.nn import FlopCountAnalysis

model_php_raw = AutoModel.from_pretrained('src/php_raw/php_codebert_model')

def main():
    print(f"cuda.is_available(): {torch.cuda.is_available()}")
    print(f"transformers 版本 {transformers.__version__}")

    show_model_info(model_php_raw)

def show_model_info(model):

    name = model.__class__.__name__
    print(f"Model name: {name}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"总参数量: {total_params / 1e6:.2f}M")

    batch_size = 1
    seq_length = 512
    dummy_input_ids = torch.ones(batch_size, seq_length, dtype=torch.long)

    flops = FlopCountAnalysis(model, (dummy_input_ids,))

    print(f"FLOPs: {flops.total() / 1e9:.2f} GFLOPs")


if __name__ == "__main__":
    main()

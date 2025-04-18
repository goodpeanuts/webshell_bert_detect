import torch
import transformers


def main():
    print(torch.cuda.is_available())
    print(transformers.__version__)


if __name__ == "__main__":
    main()

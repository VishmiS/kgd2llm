import torch

print("PyTorch version:", torch.__version__)

if torch.cuda.is_available():
    print("CUDA is available!")
    print("GPU Device Name:", torch.cuda.get_device_name(0))
    # Simple tensor operation on GPU
    x = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    y = x * 2
    print("Tensor on GPU:", y)
else:
    print("CUDA not available. Running on CPU.")

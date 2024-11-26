import torch

# Check if CUDA (GPU) is available
cuda_available = torch.cuda.is_available()
print(f"CUDA available: {cuda_available}")

if cuda_available:
    # Get the number of GPUs
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs: {num_gpus}")

    # Get the name of each GPU
    for i in range(num_gpus):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

    # Set the default device
    print(f"Current device: {torch.cuda.current_device()}")
else:
    print("GPU not available. Using CPU instead.")


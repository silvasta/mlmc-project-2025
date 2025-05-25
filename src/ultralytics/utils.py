import torch


# load cuda device and print some information
def load_device_print_information():
    # def load_device_print_information ()-> String :
    if torch.cuda.is_available():
        device = "cuda"
        print("CUDA is available! PyTorch can use the GPU.")
        print(f"Device count: {torch.cuda.device_count()}")
        print(f"Device name: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        print("CUDA is not available. PyTorch cannot use the GPU.")
    # summary with memory
    print()
    print(f"Using device: {device}")
    mem = torch.cuda.get_device_properties(0).total_memory
    print(f"Total device memory: {mem} B")
    print(f"Total device memory: {round(mem / 2**10, 3)} KB")
    print(f"Total device memory: {round(mem / 2**20, 3)} MB")
    print(f"Total device memory: {round(mem / 2**30, 3)} GB")
    print()

    return device

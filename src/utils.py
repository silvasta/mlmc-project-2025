import torch


# load cuda device and print some information
def load_device_print_information(spacing=True, crash=True):
    if torch.cuda.is_available():
        device = "cuda"
        print("CUDA is available! PyTorch can use the GPU.")
        print(f"Device count: {torch.cuda.device_count()}")
        print(f"Using device: {device}")
        mem = torch.cuda.get_device_properties(0).total_memory
        print(f"Total device memory: {round(mem / 2**30, 3)} GB")
    else:
        device = "cpu"
        print("CUDA is not available. PyTorch uses CPU.")

    return device


# very simple function that just prints the content...
# ...of the ultralytics settings (usually) located at:
# $HOME/.config/Ultralytics/settings.json
def print_ultralytics_settings():
    from ultralytics import settings

    print(settings)

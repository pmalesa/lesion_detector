import torch


def show_available_devices():
    print(f"Cuda available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Current device index: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")

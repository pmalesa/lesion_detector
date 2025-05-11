from tensorflow.config import list_physical_devices


def show_available_devices():
    """
    Prints the list of available devices.
    """

    message = f"\n  Number of GPUs available: {len(list_physical_devices('GPU'))}"
    message += "\n  Available devices:"
    for device in list_physical_devices():
        message += f"\n   - {device}"
    print(message)

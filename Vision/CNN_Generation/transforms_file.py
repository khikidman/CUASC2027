import torch

def extract_green_channel(image_tensor):
    # Assuming image_tensor shape is (3, H, W) and green is channel 1
    return image_tensor[1:2, :, :]
from ultralytics import YOLO
import cv2 as cv
import vpi
import numpy as np
import torch
from vpi_without_cuda import return_points2

img = cv.imread("../Media/rotated_test1.jpg")
model = YOLO("best.pt")

"""
vpi_frame = vpi.asimage(np.ascontiguousarray(img), vpi.Format.RGB8)
vpi_frame = vpi_frame.rescale((1920, 1920), interp=vpi.Interp.LINEAR, backend=vpi.Backend.CUDA)


vpi_image_input = vpi_frame.rescale((640, 640), interp=vpi.Interp.LINEAR, backend=vpi.Backend.CUDA)
results = None
with vpi_image_input.rlock_cuda() as cuda_buffer:
    # Zero-copy wrap into a PyTorch GPU tensor (Shape: H, W, C)
    tensor_hwc = torch.as_tensor(cuda_buffer, device="cuda")

    # Convert HWC -> BCHW and normalize to [0.0, 1.0] for YOLO
    tensor_bchw = tensor_hwc.permute(2, 0, 1).unsqueeze(0).float().div_(255.0)

    # Run YOLO inference directly on the GPU tensor
    results = model(tensor_bchw, conf=0.74)
    

grayscaled_input_image = vpi.Image(vpi_frame.size, vpi.Format.U8)
with vpi.Backend.CUDA:
    vpi.mixchannels([vpi_frame], [grayscaled_input_image], [1], [0])

np_array_img = vpi_frame.cpu()
"""
results = model(img, conf=0.74)
return_points2(results, img[:, :, 1])

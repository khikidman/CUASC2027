# Make sure to intsall YOLO, OpenCV, ans NumPy
from ultralytics import YOLO
import cv2 as cv
import numpy as np
from pathlib import Path
import math
import torch
import torchvision.transforms.functional as TF

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

model = YOLO("Vision/best.pt").to(device)

"""Custom CNN"""
exported_program = torch.export.load("Vision/team_air_gcp_localization_protoype.pt2")
cnn = exported_program.module()

"""This is the final file with all the final numbers hardcoded"""

"""Load template matching files into GPUMat"""
directory = Path('Vision/Numbers')
files = sorted([f for f in directory.iterdir() if f.is_file() and f.suffix.lower() == ".jpg"])
templates = []
for entry in files:
    gpu_temp = cv.cuda_GpuMat()
    template = cv.imread(entry)[:, :, 1]
    gpu_temp.upload(template)
    templates.append(gpu_temp)

"""Find Squares Algo: CPU"""
# Extract corners from the square (YOLO26 outputs a segmentation maks, not the points themselves)
# OpenCV has built in libraries for this
def extract_corners(polygon, shape):
    # We first check if image is fully captured
    # Image shape
    h, w = shape
    # Split into x and y arrays
    x = polygon[:, 0]
    y = polygon[:, 1]

    # We don't consider if boundaries are not in (1-2*threshold)
    threshold = 0.01

    # Check
    valid = (
        threshold*w <= x.min() and x.max() < (1-threshold)*w
        and threshold*h < y.min() and y.max() < (1-threshold)*h
    )

    if not valid:
        print("Captured image to close to borders, skipping")
        return None

    contour = np.asarray(polygon, dtype=np.float32).reshape(-1, 1, 2)
    perimeter = cv.arcLength(contour, True)

    for epsilon_factor in [0.01, 0.02, 0.03, 0.05]:
        # RDP Algo
        approx = cv.approxPolyDP(contour, epsilon_factor * perimeter, True)
        if len(approx) == 4:
            if (epsilon_factor == 0.05):
                # Should end earlier; if not, images are overly noisy
                print("Received high noise")

            return approx.reshape(4, 2)

    print("Captured non-square; likely partially captured")
    return None

# Actually fix the image depending on the perspective of the square, and return a rectified image
"""CPU"""
def rectify_target(image, corners):
    # Sort corners cyclically around their centroid
    center = corners.mean(axis=0)

    angles = np.arctan2(
        corners[:, 1] - center[1],
        corners[:, 0] - center[0]
    )

    corners = corners[np.argsort(angles)]

    # Map sorted points to spots on final image
    destination = np.array([
        [0, 0],
        [255, 0],
        [255, 255],
        [0, 255]
    ], dtype=np.float32)

    # Calculate homography matrix for transformation
    H = cv.getPerspectiveTransform(corners.astype(np.float32),destination)
    """GPU"""
    dst = cv.cuda.GpuMat(256, 256, img.type())
    cv.cuda.warpPerspective(image, dst, H, (256, 256), flags=cv.INTER_LINEAR)
    return dst
    

"""Display Numbers from Square Algo"""
"""IDK if CPU or GPU"""
# Find where the white and black triangles are located
def analyze_squares(image, padding=10):
    """
    Explanation of params:
    Image shape (256, 256) = (size, size) GPUMAT
    Size: sampling square side length
    Padding: distance from the corresponding outer edge

    Credit to Khi for the idea
    """

    N = 256
    size_sq = 48
    center = (256 - size_sq) // 2

    # Division
    regions = {
        "top": image[
            padding:padding + size_sq,
            center:center + size_sq
        ],

        "bottom": image[
            N - padding - size_sq:N - padding,
            center:center + size_sq
        ],

        "left": image[
            center:center + size_sq,
            padding:padding + size_sq
        ],

        "right": image[
            center:center + size_sq,
            N - padding - size_sq:N - padding
        ]
    }

    # Calculate sum for each view
    # Should be efficient np on cuda
    sums = {
        name: cv.cuda.sum(region)
        for name, region in regions.items()
    }

    horizontal = sums["left"] + sums["right"]
    vertical = sums["top"] + sums["bottom"]

    if horizontal > vertical: 
        # Black triangles are on left and right
        axis = "horizontal"
        candidates = ("left", "right")
    else:
        # Black triangles are on top and bottom
        axis = "vertical"
        candidates = ("top", "bottom")

    # The numbered triangle should be darker, so lower green value
    number_side = min(candidates, key=lambda x: sums[x])

    return axis, number_side

# This returns a np array of shape (256, 256/2) or (256, heigh/2) = (256, 128) currently containing number
"""GPU"""
"""PRBLY HAVE TO CONVERT FROM GPUMAT TO TORCH TENSOR"""
"""THIS IS WHAT GOES INTO IMAGE CLASSIFICATION MODEL"""
def retrieve_numbers(rectified_image):
    axis, number_side = analyze_squares(rectified_image)

    # half will be GPUMat
    half = None
    # Shape of rectified_image is (256, 256)
    if axis == "horizontal":
        # Slice from the middle of the 256 to the ends
        half = rectified_image[:, 128:] if number_side == "right" else rectified_image[:, :128]
        if number_side == "left":
            half = cv.cuda.flip(half, flipCode = 1)
    else:
        # Slice from the middle of the 256 to the ends
        half = rectified_image[128:, :] if number_side == "bottom" else rectified_image[:128, :]
        
        if number_side == "top":
            half = cv.cuda.rotate(half, rotateCode=cv.ROTATE_90_CLOCKWISE)
        else:
            half = cv.cuda.rotate(half, rotateCode=cv.ROTATE_90_COUNTERCLOCKWISE)

    num = half[
        74 : 168, 
        40 : 120
    ] 
    return num

"""Classification with template matching"""
def conf_func(diff, val):
    return ((2 / (1 + math.exp(-20*diff))) - 1) * ((2 / (1 + math.exp(-4.7*val))) - 1)

# Run template matching
def template_match(img):
    l = []
    for template in templates:
        method = cv.TM_CCOEFF_NORMED
        matcher = cv.cuda.createTemplateMatching(img.type(), method)
        result_gpu = matcher.match(img, template)

        # Download the matrix to parse value limits
        result_cpu = result_gpu.download()
        max_val = cv.minMaxLoc(result_cpu)[1]
        l.append(max_val)

    arr = sorted(l)
    # num, conf
    return np.argmax(np.array(l)), conf_func(arr[-1] - arr[-2], arr[-1])


def return_points(results, image):
    for i, result in enumerate(results):
        if (result.masks == None):
                print(f"No results found in image {i}")
                continue

        for polygon in result.masks.xy:
            corners = extract_corners(polygon, image.shape)
            if corners is not None:
                centroid = np.mean(corners, axis=0) # (x, y) in pixels

                gpu_img = cv.cuda_GpuMat()
                gpu_img.upload(image)
                gpu_dst = rectify_target(gpu_img, corners)

                num_img = retrieve_numbers(gpu_dst) # num_img is GPUMat

                num, conf = template_match(num_img)

                if (conf < 0.85):
                    ctx = num_img.__cuda_array_interface__
                    # This creates a torch.Tensor pointing directly to the GPU memory buffer
                    tensor = torch.as_tensor(ctx, device='cuda')

                    # Add the explicit channel dimension to achieve (1, 1, 94, 80)
                    tensor = tensor.unsqueeze(0).unsqueeze(0)
                    num_tensor = tensor.to(dtype=torch.float32).div(255.0)
                    with torch.inference_mode():
                        cnn_output = cnn(num_tensor)
                        if (torch.max(torch.softmax(cnn_output, dim=1)).item() > conf):
                            num = torch.argmax(cnn_output, dim=1).item()
                            conf = torch.max(torch.softmax(cnn_output, dim=1)).item()

                if (conf < 0.7):
                    # To avoid false positives
                    print("Cannot reliably detect image; skipping")
                    continue

                print(f"Found number {num} at pixel coordinates {centroid}")
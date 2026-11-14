from pathlib import Path
import math
import cv2 as cv
import numpy as np
import torch
import torchvision.transforms.functional as TF
from ultralytics import YOLO

"""Converted to OpenCV CUDA"""

assert cv.cuda.getCudaEnabledDeviceCount() > 0, (
    "OpenCV CUDA is not enabled. On JetPack 6.2.1, build OpenCV from source with "
    "-D WITH_CUDA=ON -D CUDA_ARCH_BIN=8.7 (Orin Nano Ampere SM 8.7)."
)
cv.cuda.setDevice(0)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = YOLO("Vision/best.pt")

"""Custom CNN"""
exported_program = torch.export.load("Vision/team_air_gcp_localization_protoype.pt2")
cnn = exported_program.module()

"""Load and upload template matching files to GPU once at startup"""
directory = Path("Vision/Numbers")
files = sorted([f for f in directory.iterdir() if f.is_file() and f.suffix.lower() == ".jpg"])
gpu_templates: list[cv.cuda_GpuMat] = []
for entry in files:
    # Ensure contiguous single-channel green array before uploading to GpuMat
    template = np.ascontiguousarray(cv.imread(str(entry))[:, :, 1])
    gpu_tmpl = cv.cuda_GpuMat()
    gpu_tmpl.upload(template)
    gpu_templates.append(gpu_tmpl)

# Reusable CUDA template matcher (avoids per-call handle initialization)
template_matcher = cv.cuda.createTemplateMatching(cv.CV_8UC1, cv.TM_CCOEFF_NORMED)


"""Find Squares Algo (CPU - small 1D polygon contour operations)"""
def extract_corners(polygon: np.ndarray, shape: tuple[int, ...]) -> np.ndarray | None:
    h, w = shape[:2]
    x = polygon[:, 0]
    y = polygon[:, 1]

    threshold = 0.01
    valid = (
        threshold * w <= x.min()
        and x.max() < (1 - threshold) * w
        and threshold * h <= y.min()
        and y.max() < (1 - threshold) * h
    )

    if not valid:
        print("Captured image too close to borders, skipping")
        return None

    contour = np.asarray(polygon, dtype=np.float32).reshape(-1, 1, 2)
    perimeter = cv.arcLength(contour, True)

    for epsilon_factor in (0.01, 0.02, 0.03, 0.05):
        approx = cv.approxPolyDP(contour, epsilon_factor * perimeter, True)
        if len(approx) == 4:
            if epsilon_factor == 0.05:
                print("Received high noise")
            return approx.reshape(4, 2)

    print("Captured non-square; likely partially captured")
    return None

"""Performs perspective transform with CUDA"""
def rectify_target(gpu_image: cv.cuda_GpuMat, corners: np.ndarray) -> cv.cuda_GpuMat:
    center = corners.mean(axis=0)
    angles = np.arctan2(corners[:, 1] - center[1], corners[:, 0] - center[0])
    corners = corners[np.argsort(angles)]

    destination = np.array(
        [[0, 0], [255, 0], [255, 255], [0, 255]],
        dtype=np.float32,
    )

    # 3x3 homography is computed on CPU; warp runs on CUDA
    H = cv.getPerspectiveTransform(corners.astype(np.float32), destination)
    return cv.cuda.warpPerspective(gpu_image, H, (256, 256))


"""Display Numbers from Square Algo (GPU GpuMat operations)"""
def analyze_squares(gpu_rectified: cv.cuda_GpuMat, padding: int = 10) -> tuple[str, str]:
    N = 256
    size_sq = 48
    center = (N - size_sq) // 2

    # GpuMat does not support NumPy [y1:y2, x1:x2] slicing in Python;
    # rowRange().colRange() creates zero-copy GPU ROI views.
    regions = {
        "top": gpu_rectified.rowRange(
            padding, padding + size_sq).colRange(
            center, center + size_sq
        ),
        "bottom": gpu_rectified.rowRange(
            N - padding - size_sq, N - padding).colRange(
            center, center + size_sq
        ),
        "left": gpu_rectified.rowRange(
            center, center + size_sq).colRange(
            padding, padding + size_sq
        ),
        "right": gpu_rectified.rowRange(
            center, center + size_sq).colRange(
            N - padding - size_sq, N - padding
        ),
    }

    # Compute region sums directly on GPU via absSum (uint8 values >= 0)
    sums = {
        name: cv.cuda.absSum(region)[0] for name, region in regions.items()
    }

    horizontal = sums["left"] + sums["right"]
    vertical = sums["top"] + sums["bottom"]

    if horizontal > vertical:
        axis = "horizontal"
        candidates = ("left", "right")
    else:
        axis = "vertical"
        candidates = ("top", "bottom")

    number_side = min(candidates, key=lambda x: sums[x])
    return axis, number_side


def retrieve_numbers(gpu_rectified: cv.cuda_GpuMat) -> cv.cuda_GpuMat:
    axis, number_side = analyze_squares(gpu_rectified)

    if axis == "horizontal":
        gpu_half = (
            gpu_rectified.colRange(128, 256)
            if number_side == "right"
            else gpu_rectified.colRange(0, 128)
        )
        if number_side == "left":
            gpu_half = cv.cuda.flip(gpu_half, -1)
    else:
        gpu_half = (
            gpu_rectified.rowRange(128, 256)
            if number_side == "bottom"
            else gpu_rectified.rowRange(0, 128)
        )
        # Exact 90-degree rotations on GPU via transpose + flip (zero interpolation)
        gpu_transposed = cv.cuda.transpose(gpu_half)
        if number_side == "top":
            # Equivalent to cv.ROTATE_90_CLOCKWISE
            gpu_half = cv.cuda.flip(gpu_transposed, 1)
        else:
            # Equivalent to cv.ROTATE_90_COUNTERCLOCKWISE
            gpu_half = cv.cuda.flip(gpu_transposed, 0)

    # Clone the 94x80 ROI so downstream matchTemplate receives contiguous GPU memory
    return gpu_half.rowRange(74, 168).colRange(40, 120).clone()


"""Classification with CUDA template matching"""
def conf_func(diff: float, val: float) -> float:
    return ((2 / (1 + math.exp(-20 * diff))) - 1) * (
        (2 / (1 + math.exp(-4.7 * val))) - 1
    )


"""Run Template Macthing on CUDA"""
def template_match(gpu_img: cv.cuda_GpuMat) -> tuple[int, float]:
    scores = []
    for gpu_tmpl in gpu_templates:
        gpu_res = template_matcher.match(gpu_img, gpu_tmpl)
        _, max_val, _, _ = cv.cuda.minMaxLoc(gpu_res)
        scores.append(max_val)

    arr = sorted(scores)
    best_idx = int(np.argmax(scores))
    return best_idx, conf_func(arr[-1] - arr[-2], arr[-1])


def return_points(results, image: np.ndarray) -> None:
    # Upload input frame to GPU once per frame
    gpu_frame = cv.cuda_GpuMat()
    gpu_frame.upload(np.ascontiguousarray(image))

    # Extract green channel on GPU if a 3-channel BGR image was passed
    if gpu_frame.channels() == 3:
        gpu_green = cv.cuda.split(gpu_frame)[1]
    else:
        gpu_green = gpu_frame

    for i, result in enumerate(results):
        if result.masks is None:
            print(f"No results found in image {i}")
            continue

        for polygon in result.masks.xy:
            corners = extract_corners(polygon, image.shape)
            if corners is None:
                continue

            centroid = np.mean(corners, axis=0)  # (x, y) in pixels
            gpu_rectified = rectify_target(gpu_green, corners)
            gpu_num_img = retrieve_numbers(gpu_rectified)
            num, conf = template_match(gpu_num_img)

            if conf < 0.85:
                # Download only the tiny 94x80 patch if CNN fallback is needed
                num_img_cpu = gpu_num_img.download()
                num_tensor = TF.to_tensor(num_img_cpu).unsqueeze(0).to(device)
                with torch.inference_mode():
                    cnn_output = cnn(num_tensor)
                    probs = torch.softmax(cnn_output, dim=1)
                    cnn_conf, cnn_num = torch.max(probs, dim=1)
                    if cnn_conf.item() > conf:
                        num = int(cnn_num.item())
                        conf = float(cnn_conf.item())

            if conf < 0.7:
                print("Cannot reliably detect image; skipping")
                continue

            print(f"Found number {num} at pixel coordinates {centroid}")
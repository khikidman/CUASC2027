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

"""Load template matching files"""
directory = Path('Vision/Numbers')
l = []
files = sorted([f for f in directory.iterdir() if f.is_file() and f.suffix.lower() == ".jpg"])

"""Find Squares Algo"""
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
    return cv.warpPerspective(image, H, (256, 256))

"""Display Numbers from Square Algo"""
# Find where the white and black triangles are located
def analyze_squares(image, padding=10):
    """
    Explanation of params:
    Image shape (256, 256) = (size, size)
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

    # Calculate mean for each view
    # Should be efficient np on cuda
    means = {
        name: region.mean()
        for name, region in regions.items()
    }

    horizontal = means["left"] + means["right"]
    vertical = means["top"] + means["bottom"]

    if horizontal > vertical: 
        # Black triangles are on left and right
        axis = "horizontal"
        candidates = ("left", "right")
    else:
        # Black triangles are on top and bottom
        axis = "vertical"
        candidates = ("top", "bottom")

    # The numbered triangle should be darker, so lower green value
    number_side = min(candidates, key=lambda x: means[x])

    return axis, number_side

# This returns a np array of shape (256, 256/2) or (256, heigh/2) = (256, 128) currently containing number
"""THIS IS WHAT GOES INTO IMAGE CLASSIFICATION MODEL"""
def retrieve_numbers(rectified_image):
    axis, number_side = analyze_squares(rectified_image)

    half = None
    # Shape of rectified_image is (256, 256)
    if axis == "horizontal":
        # Slice from the middle of the 256 to the ends
        half = rectified_image[:, 128:] if number_side == "right" else rectified_image[:, :128]
        if number_side == "left":
            half = cv.flip(half, -1)
    else:
        # Slice from the middle of the 256 to the ends
        half = rectified_image[128:, :] if number_side == "bottom" else rectified_image[:128, :]
        
        if number_side == "top":
            half = cv.rotate(half, cv.ROTATE_90_CLOCKWISE)
        else:
            half = cv.rotate(half, cv.ROTATE_90_COUNTERCLOCKWISE)

    num = half[
        74 : 168, 
        40 : 120
    ] 
    return num

"""Classification with template matching"""
def conf_func(diff, val):
    return ((2 / (1 + math.exp(-20*diff))) - 1) * (1 / (1 + math.exp(-3.7*val)))

# Run template matching
def template_match(img):
    # In the final file, we should load the files into cuda before run
    for entry in files:
        template = cv.imread(entry)[:, :, 1]
        res = cv.matchTemplate(img, template, cv.TM_CCOEFF_NORMED)
        l.append(cv.minMaxLoc(res)[1])

    arr = sorted(l)
    # num, conf
    return np.argmax(np.array(l)), 1 - conf_func(arr[-1] - arr[-2], arr[-1])





"""DEBUGGING"""
# Classification with display
def find_numbers(results, image):
    for i, result in enumerate(results):
        # Checks if there are results
        if (result.masks == None):
            print(f"No results found in image {i}")
            continue

        for polygon in result.masks.xy:
            corners = extract_corners(polygon, image.shape)
            if corners is not None:
                rectified_image = rectify_target(image, corners)
                num_img = retrieve_numbers(rectified_image)
                num, conf = template_match(num_img)

                # Create new image with information
                res_img = cv.copyMakeBorder(
                    num_img, 0, 0, 0, 100, 
                    borderType=cv.BORDER_CONSTANT, 
                    value= 0
                )

                mod = "Tem"
                if (conf < 0.8):
                    num_tensor = TF.to_tensor(num_img).unsqueeze(0).to(device)
                    with torch.inference_mode():
                        cnn_output = cnn(num_tensor)
                        if (torch.max(torch.softmax(cnn_output, dim=1)).item() > conf):
                            num = torch.argmax(cnn_output, dim=1).item()
                            conf = torch.max(torch.softmax(cnn_output, dim=1)).item()
                            mod = "CNN"
                
                res_img = cv.putText(res_img, f"Number: {num} \n{mod} Conf: \n{conf:4f}", (90, 20), cv.FONT_HERSHEY_PLAIN, 1, 255, 1)

                cv.imshow("Classification", res_img)
                cv.waitKey(0)
                cv.destroyWindow("Classification")


# If it is 1, you can use an image in local directory; change to 0 to use webcam
if (1):
    # Test on an image in local directory
    img = cv.imread("Vision/Media/test_hard.jpg")
    results = model(img, conf=0.74) # Possibly change confidence value
    plot = results[0].plot()
    cv.imshow("Results", plot)
    cv.waitKey(0)

    find_numbers(results, img[:, :, 1]) # Extract just the greens
    #display_image(results, img[:, :, 1])

else:
    # Test on a video stream from webcam
    cap = cv.VideoCapture(0) # Should be the main camera, if you have continuinty cam, it will be on your phone (try 1 or -1 if no feed appears)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
    
        # if frame is read correctly, ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        results = model(frame, verbose=False, conf=0.74)
        annotated_frame = results[0].plot()
        cv.imshow('frame', annotated_frame)

        key = cv.waitKey(1)
        if key == ord(' '):
            find_numbers(results, frame[:, :, 1])

        elif key == ord('q'):
            exit()
    
    cap.release()

cv.destroyAllWindows()
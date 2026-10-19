# Make sure to intsall YOLO and OpenCV
from ultralytics import YOLO
import cv2 as cv
import numpy as np

model = YOLO("Vision/best.pt").to("mps") # Set to 'cpu' if you have Windows or non-M Mac, or 'cuda' if you have Nvidia GPU

"""
This file is for prototyping. Once we verify final values, we should create a final file with fixed values
Here are the global variables that allow for easy prototyping:
"""
# Size: Each segmentation mask is warped by homography to a height x width square
# Would not recommend rectangular regions as then we would get inconsistent halves and would need to apply transformation, which is expensive
height = 256
width = height
# The other three experimental numbers are the padding and size_sq in analyze_squares() and the ratios in retrieve_numbers()

"""Find Squares Algo"""
# Extract corners from the square (YOLO26 outputs a segmentation maks, not the points themselves)
# OpenCV has built in libraries for this
def extract_corners(polygon):
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
        [height - 1, 0],
        [height - 1, width - 1],
        [0, width - 1]
    ], dtype=np.float32)

    # Calculate homography matrix for transformation
    H = cv.getPerspectiveTransform(corners.astype(np.float32),destination)
    return cv.warpPerspective(image, H, (height, width))

# For testing purposes
# Displays the square returned by the two earlier functions
def display_image(results, img):
    for i, result in enumerate(results):
        # Checks if there are results
        if (result.masks == None):
            print(f"No results found in image {i}")
            continue
            
        for polygon in result.masks.xy:
            corners = extract_corners(polygon)
            if corners is not None:
                rectified_image = rectify_target(img, corners)
                cv.imshow("Rectified Image", rectified_image)
                cv.waitKey(0)


"""Display Numbers from Square Algo"""
# Find where the white and black triangles are located
def analyze_squares(image, padding=10):
    """
    Image shape (256, 256) = (size, size)
    Size: sampling square side length
    Padding: distance from the corresponding outer edge
    Credit to Khi for the idea
    """

    # Height, width, works for variable sizes in case we need to change
    N_h = image.shape[0]
    N_w = image.shape[1]
    size_sq_h = round(height * (48/256))
    size_sq_w = round(width * (48/256))
    center_h = (N_h - size_sq_h) // 2
    center_w = (N_w - size_sq_w) // 2

    # Division
    regions = {
        "top": image[
            padding:padding + size_sq_h,
            center_w:center_w + size_sq_w
        ],

        "bottom": image[
            N_h - padding - size_sq_h:N_h - padding,
            center_w:center_w + size_sq_w
        ],

        "left": image[
            center_h:center_h + size_sq_h,
            padding:padding + size_sq_w
        ],

        "right": image[
            center_h:center_h + size_sq_h,
            N_w - padding - size_sq_w:N_w - padding
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

# This returns a np array of shape (height, width/2) or (width, heigh/2) = (256, 128) currently containing number
def retrieve_numbers(rectified_image):
    axis, number_side = analyze_squares(rectified_image)

    half = None
    
    # The rotations are why rectangular regions are problematic, but we can fix using this converting it to square
    size = (height + width) // 2
    rectified_image = cv.resize(rectified_image, (size, size), interpolation=cv.INTER_NEAREST)

    # Shape of rectified_image is (size, size)
    if axis == "horizontal":
        # Slice from the middle of the width to the ends
        half = rectified_image[:, size // 2:] if number_side == "right" else rectified_image[:, :size // 2]
        if number_side == "left":
            half = cv.flip(half, -1)
    else:
        # Slice from the middle of the height to the ends
        half = rectified_image[size // 2:, :] if number_side == "bottom" else rectified_image[:size // 2, :]
        
        if number_side == "top":
            half = cv.rotate(half, cv.ROTATE_90_CLOCKWISE)
        else:
            half = cv.rotate(half, cv.ROTATE_90_COUNTERCLOCKWISE)

    # ROUGH calculations, can be changed later
    num = half[
        round(size*(74/256)) : round(size*(168/256)), 
        round(size*(40/256)) : round(size*(120/256))
    ] 
    return num

# This just displays the results of retrieve_numbers()
def display_squares(results, image):
    for i, result in enumerate(results):
        # Checks if there are results
        if (result.masks == None):
            print(f"No results found in image {i}")
            continue

        for polygon in result.masks.xy:
            corners = extract_corners(polygon)
            if corners is not None:
                rectified_image = rectify_target(image, corners)
                num = retrieve_numbers(rectified_image)
                cv.imshow("Number", num)
                cv.waitKey(0)


# If it is 1, you can use an image in local directory; change to 0 to use webcam
if (1):
    # Test on an image in local directory
    img = cv.imread("Vision/Media/rotated_test1.jpg")
    results = model(img, conf=0.74) # Possibly change confidence value
    display_squares(results, img[:, :, 1]) # Extract just the greens
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

        results = model(frame, verbose=False, conf=0.6)
        annotated_frame = results[0].plot()
        cv.imshow('frame', annotated_frame)
        if cv.waitKey(1) == ord('q'):
            # display_image(results, frame[:, :, 1])
            display_squares(results, frame[:, :, 1])
            break

    cap.release()

cv.destroyAllWindows()

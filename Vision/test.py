# Make sure to intsall YOLO and OpenCV
from ultralytics import YOLO
import cv2 as cv
import numpy as np

model = YOLO("Vision/best.pt").to("mps") # Set to cpu if you have Windows

# Extract corners from the square (YOLO26 outputs a segmentation maks, not the points themselves)
def extract_corners(polygon):
    contour = np.asarray(polygon, dtype=np.float32).reshape(-1, 1, 2)
    perimeter = cv.arcLength(contour, True)

    for epsilon_factor in [0.01, 0.02, 0.03, 0.05]:
        approx = cv.approxPolyDP(contour, epsilon_factor * perimeter, True)
        if len(approx) == 4:
            return approx.reshape(4, 2)

    return None

# Actually fix the image depending on the perspective of the square, and return a rectified image
def rectify_target(image, corners, size=256):
    # Sort corners cyclically around their centroid
    center = corners.mean(axis=0)

    angles = np.arctan2(
        corners[:, 1] - center[1],
        corners[:, 0] - center[0]
    )

    corners = corners[np.argsort(angles)]

    destination = np.array([
        [0, 0],
        [size - 1, 0],
        [size - 1, size - 1],
        [0, size - 1]
    ], dtype=np.float32)

    H = cv.getPerspectiveTransform(corners.astype(np.float32),destination)
    return cv.warpPerspective(image, H, (size, size))

# For visual testing purposes
def display_image(results, img):
    for result in results:
        for polygon in result.masks.xy:
            corners = extract_corners(polygon)
            if corners is not None:
                rectified_image = rectify_target(img, corners)
                cv.imshow("Rectified Image", rectified_image)
                cv.waitKey(0)

if (1):
    # Test on an image
    img = cv.imread("Vision/hundredfeet.jpg")
    results = model(img, conf=0.6)
    display_image(results, img)
else:
    # Test on a video stream
    cap = cv.VideoCapture(0) # Should be the main camera, if you have continuinty cam, it will be on your phone
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
    
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        results = model(frame, verbose=False, conf=0.6)
        annotated_frame = results[0].plot()
        cv.imshow('frame', annotated_frame)
        if cv.waitKey(1) == ord('q'):
            display_image(results, frame)
            break

    cap.release()

cv.destroyAllWindows()

import cv2
import os
import time
import datetime
import json

from ultralytics import YOLO


# ============================================================
# Configuration
# ============================================================

OUTPUT_DIR = "ml_images"

PHOTO_INTERVAL = 5.0

CAMERA_SENSOR_ID = 0
CAMERA_WIDTH = 4056
CAMERA_HEIGHT = 3040
CAMERA_FRAMERATE = 10

# GStreamer flip:
# 0 = none
# 2 = 180 degrees
FLIP_METHOD = 0

JPEG_QUALITY = 95


# ============================================================
# YOLO Configuration
# ============================================================

MODEL_PATH = "yolo26.engine"

# Minimum confidence for a detection to be recorded
CONFIDENCE_THRESHOLD = 0.25

# Directory for annotated images
ANNOTATED_DIR = os.path.join(
    OUTPUT_DIR,
    "annotated"
)

# Directory for detection metadata
METADATA_DIR = os.path.join(
    OUTPUT_DIR,
    "metadata"
)


# ============================================================
# Create directories
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(ANNOTATED_DIR, exist_ok=True)
os.makedirs(METADATA_DIR, exist_ok=True)


# ============================================================
# GStreamer pipeline
# ============================================================

def gstreamer_pipeline():

    return (
        f"nvarguscamerasrc "
        f"sensor-id={CAMERA_SENSOR_ID} "
        f"wbmode=1 ! "
        f"video/x-raw(memory:NVMM),"
        f"width=(int){CAMERA_WIDTH},"
        f"height=(int){CAMERA_HEIGHT},"
        f"framerate=(fraction){CAMERA_FRAMERATE}/1 ! "
        f"nvvidconv flip-method={FLIP_METHOD} ! "
        f"video/x-raw,format=(string)BGRx ! "
        f"videoconvert ! "
        f"video/x-raw,format=(string)BGR ! "
        f"appsink drop=true max-buffers=1 sync=false"
    )


# ============================================================
# Main
# ============================================================

def main():

    # --------------------------------------------------------
    # Load YOLO model
    # --------------------------------------------------------

    print("Loading YOLO26 TensorRT model...")

    try:

        model = YOLO(MODEL_PATH)

    except Exception as e:

        print("ERROR: Failed to load YOLO model.")
        print(e)
        return

    print("YOLO model loaded.")
    print()


    # --------------------------------------------------------
    # Open camera
    # --------------------------------------------------------

    print("Opening camera...")

    pipeline = gstreamer_pipeline()

    print("Pipeline:")
    print(pipeline)
    print()

    cam = cv2.VideoCapture(
        pipeline,
        cv2.CAP_GSTREAMER
    )

    if not cam.isOpened():

        print("ERROR: Camera failed to open.")
        return

    print("Camera opened.")

    # Give camera time to start
    time.sleep(2)


    # --------------------------------------------------------
    # Flush startup frames
    # --------------------------------------------------------

    print("Flushing camera buffer...")

    for _ in range(20):
        cam.grab()

    print("Buffer flushed.")
    print()


    # --------------------------------------------------------
    # Information
    # --------------------------------------------------------

    print("==========================================")
    print("ML IMAGE CAPTURE")
    print("==========================================")
    print(f"Resolution : {CAMERA_WIDTH} x {CAMERA_HEIGHT}")
    print(f"Interval   : {PHOTO_INTERVAL} seconds")
    print(f"YOLO Model : {MODEL_PATH}")
    print(f"Confidence : {CONFIDENCE_THRESHOLD}")
    print(f"Directory  : {OUTPUT_DIR}")
    print()
    print("Press CTRL+C to stop.")
    print("==========================================")
    print()


    image_number = 0

    next_capture = time.monotonic()


    # ========================================================
    # Capture loop
    # ========================================================

    try:

        while True:

            # ------------------------------------------------
            # Wait until scheduled capture
            # ------------------------------------------------

            now = time.monotonic()

            wait_time = next_capture - now

            if wait_time > 0:

                time.sleep(wait_time)


            # ------------------------------------------------
            # Get newest frame
            # ------------------------------------------------

            ret, frame = cam.read()

            if not ret:

                print(
                    "ERROR: Failed to read camera frame"
                )

                next_capture += PHOTO_INTERVAL
                continue


            capture_time = datetime.datetime.now()


            # ------------------------------------------------
            # Generate filename
            # ------------------------------------------------

            timestamp = (
                capture_time.strftime(
                    "%Y%m%d_%H%M%S_%f"
                )[:-3]
            )

            filename = os.path.join(
                OUTPUT_DIR,
                timestamp + ".jpg"
            )

            annotated_filename = os.path.join(
                ANNOTATED_DIR,
                timestamp + ".jpg"
            )

            metadata_filename = os.path.join(
                METADATA_DIR,
                timestamp + ".json"
            )


            # =================================================
            # YOLO INFERENCE
            # =================================================

            inference_start = time.monotonic()

            try:

                results = model.predict(
                    source=frame,
                    conf=CONFIDENCE_THRESHOLD,
                    verbose=False
                )

            except Exception as e:

                print(
                    "ERROR: YOLO inference failed:"
                )

                print(e)

                next_capture += PHOTO_INTERVAL
                continue


            inference_time = (
                time.monotonic()
                - inference_start
            )


            # =================================================
            # Process detections
            # =================================================

            detections = []

            for result in results:

                boxes = result.boxes

                for box in boxes:

                    class_id = int(
                        box.cls[0].item()
                    )

                    confidence = float(
                        box.conf[0].item()
                    )

                    # xyxy = x1, y1, x2, y2
                    coordinates = (
                        box.xyxy[0]
                        .cpu()
                        .numpy()
                        .tolist()
                    )

                    class_name = result.names[
                        class_id
                    ]

                    detections.append(
                        {
                            "class_id": class_id,
                            "class_name": class_name,
                            "confidence": confidence,
                            "bbox_xyxy": coordinates
                        }
                    )


            # =================================================
            # Save original image
            # =================================================

            success = cv2.imwrite(
                filename,
                frame,
                [
                    cv2.IMWRITE_JPEG_QUALITY,
                    JPEG_QUALITY
                ]
            )

            if not success:

                print(
                    "ERROR: Failed to save image"
                )

                next_capture += PHOTO_INTERVAL
                continue


            # =================================================
            # Save annotated image
            # =================================================

            annotated_frame = results[0].plot()

            cv2.imwrite(
                annotated_filename,
                annotated_frame,
                [
                    cv2.IMWRITE_JPEG_QUALITY,
                    JPEG_QUALITY
                ]
            )


            # =================================================
            # Save detection metadata
            # =================================================

            metadata = {

                "timestamp": (
                    capture_time.isoformat()
                ),

                "image": filename,

                "annotated_image":
                    annotated_filename,

                "image_number":
                    image_number + 1,

                "camera": {
                    "sensor_id":
                        CAMERA_SENSOR_ID,

                    "width":
                        CAMERA_WIDTH,

                    "height":
                        CAMERA_HEIGHT,

                    "framerate":
                        CAMERA_FRAMERATE
                },

                "model": MODEL_PATH,

                "confidence_threshold":
                    CONFIDENCE_THRESHOLD,

                "inference_time_seconds":
                    inference_time,

                "detections":
                    detections
            }


            with open(
                metadata_filename,
                "w"
            ) as f:

                json.dump(
                    metadata,
                    f,
                    indent=4
                )


            # =================================================
            # Print result
            # =================================================

            image_number += 1

            print(
                f"[{capture_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}] "
                f"Saved #{image_number}: "
                f"{filename}"
            )

            print(
                f"    YOLO inference: "
                f"{inference_time:.3f} sec"
            )

            if detections:

                for detection in detections:

                    print(
                        f"    DETECTED: "
                        f"{detection['class_name']} "
                        f"({detection['confidence']:.2f})"
                    )

            else:

                print(
                    "    No objects detected."
                )


            # ------------------------------------------------
            # Schedule next capture
            # ------------------------------------------------

            next_capture += PHOTO_INTERVAL


            # Prevent catch-up if processing takes too long
            if next_capture < time.monotonic():

                next_capture = (
                    time.monotonic()
                    + PHOTO_INTERVAL
                )


    except KeyboardInterrupt:

        print()
        print("Stopping capture...")


    finally:

        cam.release()

        print(
            f"Total images captured: "
            f"{image_number}"
        )

        print("Camera released.")


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":

    main()

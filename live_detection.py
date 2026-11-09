
#!/usr/bin/env python3
"""Jetson IMX219 live YOLO segmentation + square/number detection + MAVSDK telemetry."""

import argparse
import asyncio
import math
import signal
import time
from pathlib import Path

import cv2 as cv
import numpy as np
import torch
from ultralytics import YOLO
from mavsdk import System


# ---------------- Configuration ----------------

ROOT = Path(__file__).resolve().parent

DEFAULT_PIPELINE = (
    'nvarguscamerasrc sensor-id=0 ! '
    'video/x-raw(memory:NVMM), width=(int)1920, height=(int)1080, '
    'framerate=(fraction)30/1 ! '
    'nvvidconv flip-method=2 ! video/x-raw, format=(string)BGRx ! '
    'videoconvert ! video/x-raw, format=(string)BGR ! '
    'appsink drop=true max-buffers=1 sync=false'
)

latest = {
    'gps': None,
    'pos': None,
    'att': None,
    'bat': None
}


# ---------------- Square Detection ----------------

def extract_corners(polygon, frame_shape, border_fraction=0.01):
    """Return four corners, rejecting targets clipped by image borders."""

    h, w = frame_shape[:2]

    if len(polygon) < 4:
        return None

    x, y = polygon[:, 0], polygon[:, 1]

    if (
        x.min() < border_fraction * w
        or x.max() >= (1 - border_fraction) * w
        or y.min() < border_fraction * h
        or y.max() >= (1 - border_fraction) * h
    ):
        return None

    contour = np.asarray(polygon, np.float32).reshape(-1, 1, 2)
    perimeter = cv.arcLength(contour, True)

    for factor in (0.01, 0.02, 0.03, 0.05):
        approx = cv.approxPolyDP(contour, factor * perimeter, True)

        if len(approx) == 4:
            return approx.reshape(4, 2)

    return None


def rectify_target(image_green, corners, size=256):
    """Perspective-warp a square target to a 256x256 image."""

    center = corners.mean(axis=0)

    angles = np.arctan2(
        corners[:, 1] - center[1],
        corners[:, 0] - center[0]
    )

    points = corners[np.argsort(angles)].astype(np.float32)

    start = np.argmin(points[:, 0] + points[:, 1])
    points = np.roll(points, -int(start), axis=0)

    destination = np.array([
        [0, 0],
        [size - 1, 0],
        [size - 1, size - 1],
        [0, size - 1]
    ], dtype=np.float32)

    matrix = cv.getPerspectiveTransform(points, destination)

    return cv.warpPerspective(
        image_green,
        matrix,
        (size, size)
    )


# ---------------- Number Extraction ----------------

def retrieve_number(image, padding=10):
    """Identify the numbered triangle and extract its number."""

    h, w = image.shape[:2]

    side_h = round(h * 48 / 256)
    side_w = round(w * 48 / 256)

    center_h = (h - side_h) // 2
    center_w = (w - side_w) // 2

    regions = {
        'top': image[
            padding:padding + side_h,
            center_w:center_w + side_w
        ],

        'bottom': image[
            h - padding - side_h:h - padding,
            center_w:center_w + side_w
        ],

        'left': image[
            center_h:center_h + side_h,
            padding:padding + side_w
        ],

        'right': image[
            center_h:center_h + side_h,
            w - padding - side_w:w - padding
        ]
    }

    means = {
        side: crop.mean()
        for side, crop in regions.items()
    }

    horizontal = means['left'] + means['right']
    vertical = means['top'] + means['bottom']

    if horizontal > vertical:
        sides = ('left', 'right')
    else:
        sides = ('top', 'bottom')

    number_side = min(sides, key=means.get)

    if number_side == 'right':
        half = image[:, w // 2:]

    elif number_side == 'left':
        half = cv.flip(image[:, :w // 2], -1)

    elif number_side == 'top':
        half = cv.rotate(
            image[:h // 2, :],
            cv.ROTATE_90_CLOCKWISE
        )

    else:
        half = cv.rotate(
            image[h // 2:, :],
            cv.ROTATE_90_COUNTERCLOCKWISE
        )

    crop = half[
        round(h * 74 / 256):round(h * 168 / 256),
        round(w * 40 / 256):round(w * 120 / 256)
    ]

    return crop if crop.size else None


# ---------------- Template Matching ----------------

def load_templates(directory):
    """Load number templates once at startup."""

    files = sorted(directory.glob('*.jpg'))
    templates = []

    for index, path in enumerate(files):
        bgr = cv.imread(str(path))

        if bgr is not None:
            templates.append((index, bgr[:, :, 1]))

    return templates


def classify_number(number_crop, templates):
    """Find the best matching number template."""

    if number_crop is None:
        return None

    scores = []

    for label, template in templates:

        if (
            number_crop.shape[0] < template.shape[0]
            or number_crop.shape[1] < template.shape[1]
        ):
            continue

        response = cv.matchTemplate(
            number_crop,
            template,
            cv.TM_CCOEFF_NORMED
        )

        score = float(cv.minMaxLoc(response)[1])

        if math.isfinite(score):
            scores.append((label, score))

    if not scores:
        return None

    scores.sort(
        key=lambda entry: entry[1],
        reverse=True
    )

    best_label, best_score = scores[0]

    gap = (
        best_score - scores[1][1]
        if len(scores) > 1
        else 0.0
    )

    heuristic = (
        2.0 / (1.0 + math.exp(-20.0 * gap))
        - 1.0
    )

    return best_label, best_score, heuristic


# ---------------- MAVSDK Telemetry ----------------

async def telemetry_listener(name, stream):

    async for reading in stream:
        latest[name] = reading


def draw_status(image, fps, device):

    pos = latest['pos']
    att = latest['att']
    gps = latest['gps']
    bat = latest['bat']

    lines = [
        f'Inference FPS: {fps:.1f}   Device: {device}',
        f'Satellites: {gps.num_satellites if gps else "--"}'
    ]

    if pos is not None:
        lines.extend([
            f'Lat/Lon: {pos.latitude_deg:.6f}, {pos.longitude_deg:.6f}',
            f'Rel alt: {pos.relative_altitude_m:.1f} m'
        ])

    if att is not None:
        lines.append(
            f'Yaw/Pitch/Roll: '
            f'{att.yaw_deg:.1f} / '
            f'{att.pitch_deg:.1f} / '
            f'{att.roll_deg:.1f}'
        )

    if bat is not None:
        lines.append(
            f'Battery: {bat.voltage_v:.2f} V'
        )

    panel_height = 14 + 27 * len(lines)

    cv.rectangle(
        image,
        (10, 10),
        (min(image.shape[1] - 10, 820), panel_height),
        (0, 0, 0),
        -1
    )

    for i, line in enumerate(lines):

        cv.putText(
            image,
            line,
            (22, 36 + i * 27),
            cv.FONT_HERSHEY_SIMPLEX,
            0.62,
            (100, 255, 100),
            2,
            cv.LINE_AA
        )


# ---------------- Detection Annotation ----------------

def annotate(frame, result, templates):
    """Draw segmentation masks and recognized target numbers."""

    annotated = result.plot()

    if result.masks is None:
        return annotated, 0

    green = frame[:, :, 1]
    count = 0

    for polygon in result.masks.xy:

        corners = extract_corners(
            polygon,
            frame.shape
        )

        if corners is None:
            continue

        rectified = rectify_target(
            green,
            corners
        )

        crop = retrieve_number(rectified)

        match = (
            classify_number(crop, templates)
            if templates
            else None
        )

        center = corners.mean(axis=0).astype(int)

        if match is not None:
            label, score, heuristic = match
            text = f'Number: {label}  Match: {score:.2f}'

        else:
            text = 'Square detected'

        cv.polylines(
            annotated,
            [corners.astype(np.int32)],
            True,
            (0, 255, 255),
            2
        )

        cv.putText(
            annotated,
            text,
            (
                max(0, center[0] - 75),
                max(28, center[1])
            ),
            cv.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
            cv.LINE_AA
        )

        count += 1

    return annotated, count


# ---------------- Main ----------------

async def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model',
        type=Path,
        default=Path('Vision/best.pt')
    )

    parser.add_argument(
        '--numbers',
        type=Path,
        default=Path('Vision/Numbers')
    )

    parser.add_argument(
        '--serial',
        default='serial:///dev/ttyACM0:57600'
    )

    parser.add_argument(
        '--no-telemetry',
        action='store_true'
    )

    parser.add_argument(
        '--conf',
        type=float,
        default=0.74
    )

    parser.add_argument(
        '--imgsz',
        type=int,
        default=640
    )

    parser.add_argument(
        '--device',
        default='auto',
        help='auto, cpu, or 0 (CUDA GPU)'
    )

    args = parser.parse_args()

    # Validate model
    if not args.model.is_file():
        parser.error(
            f'Model not found: {args.model}'
        )

    # Load number templates
    templates = load_templates(args.numbers)

    if not templates:
        print(
            f'Warning: no JPG templates in {args.numbers}'
        )

    # Select inference device
    device = (
        '0' if torch.cuda.is_available() else 'cpu'
    ) if args.device == 'auto' else args.device

    print(f'Loading {args.model} on {device} ...')

    model = YOLO(str(args.model))

    stop = asyncio.Event()

    try:
        asyncio.get_running_loop().add_signal_handler(
            signal.SIGINT,
            stop.set
        )
    except NotImplementedError:
        pass

    tasks = []

    # Connect to Cube Orange+
    if not args.no_telemetry:

        drone = System()

        print('Connecting to ArduPilot...')

        await drone.connect(
            system_address=args.serial
        )

        async for state in drone.core.connection_state():
            if state.is_connected:
                print('ArduPilot connected')
                break

        await drone.telemetry.set_rate_gps_info(1)
        await drone.telemetry.set_rate_position(2)
        await drone.telemetry.set_rate_attitude_euler(10)
        await drone.telemetry.set_rate_battery(1)

        tasks = [
            asyncio.create_task(
                telemetry_listener(
                    'gps',
                    drone.telemetry.gps_info()
                )
            ),
            asyncio.create_task(
                telemetry_listener(
                    'pos',
                    drone.telemetry.position()
                )
            ),
            asyncio.create_task(
                telemetry_listener(
                    'att',
                    drone.telemetry.attitude_euler()
                )
            ),
            asyncio.create_task(
                telemetry_listener(
                    'bat',
                    drone.telemetry.battery()
                )
            )
        ]

    # Initialize IMX219
    cap = cv.VideoCapture(
        DEFAULT_PIPELINE,
        cv.CAP_GSTREAMER
    )

    try:

        if not cap.isOpened():
            raise RuntimeError(
                'Could not open IMX219 with GStreamer'
            )

        cv.namedWindow(
            'VTOL Detection',
            cv.WINDOW_NORMAL
        )

        cv.resizeWindow(
            'VTOL Detection',
            1280,
            720
        )

        print(
            'Running. Press Q to exit; '
            'S to save annotated frame.'
        )

        frame_id = 0
        fps = 0.0

        while not stop.is_set():

            start = time.monotonic()

            # Read camera frame
            ok, frame = await asyncio.to_thread(
                cap.read
            )

            if not ok:
                print('Camera frame read failed')
                break

            # Run YOLO inference
            result = (
                await asyncio.to_thread(
                    model.predict,
                    frame,
                    conf=args.conf,
                    imgsz=args.imgsz,
                    device=device,
                    verbose=False
                )
            )[0]

            # Classify detected squares
            annotated, count = annotate(
                frame,
                result,
                templates
            )

            # Calculate inference FPS
            elapsed = time.monotonic() - start

            fps = (
                0.8 * fps + 0.2 / elapsed
                if fps
                else 1.0 / elapsed
            )

            # Draw MAVSDK telemetry
            draw_status(
                annotated,
                fps,
                device
            )

            cv.putText(
                annotated,
                f'Valid targets: {count}',
                (22, annotated.shape[0] - 24),
                cv.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 255),
                2
            )

            # Display frame
            cv.imshow(
                'VTOL Detection',
                annotated
            )

            key = cv.waitKey(1) & 0xFF

            if key == ord('q'):
                break

            if key == ord('s'):

                dest = ROOT / (
                    f'detection_{frame_id:05d}.jpg'
                )

                cv.imwrite(
                    str(dest),
                    annotated
                )

                print(f'Saved {dest}')

            frame_id += 1

            await asyncio.sleep(0)

    finally:

        cap.release()
        cv.destroyAllWindows()

        for task in tasks:
            task.cancel()

        if tasks:
            await asyncio.gather(
                *tasks,
                return_exceptions=True
            )

        print('Shutdown complete')


if __name__ == '__main__':
    asyncio.run(main())


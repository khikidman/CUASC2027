import asyncio
import cv2
import os
import time
import datetime
import signal

import piexif
from mavsdk import System

print(cv2.__version__)

# ============================================================
# Configuration
# ============================================================

OUTPUT_DIR = "ml_images"

PHOTO_INTERVAL = 5.0

CAMERA_SENSOR_ID = 0
CAMERA_WIDTH = 4056
CAMERA_HEIGHT = 3040
CAMERA_FRAMERATE = 10

# 0 = no rotation
# 2 = 180 degrees
CAMERA_FLIP_METHOD = 0

JPEG_QUALITY = 95

# Cube Orange+ USB connection
SERIAL_ADDRESS = "serial:///dev/ttyACM0:115200"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# Shared telemetry
# ============================================================

latest = {
    "gps": None,
    "pos": None,
    "att": None,
    "bat": None,
}


# ============================================================
# Camera pipeline
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
        f"nvvidconv flip-method={CAMERA_FLIP_METHOD} ! "
        f"video/x-raw,format=(string)BGRx ! "
        f"videoconvert ! "
        f"video/x-raw,format=(string)BGR ! "
        f"appsink drop=true max-buffers=1 sync=false"
    )


# ============================================================
# Telemetry listeners
# ============================================================

async def gps_listener(drone):
    async for data in drone.telemetry.gps_info():
        latest["gps"] = data


async def position_listener(drone):
    async for data in drone.telemetry.position():
        latest["pos"] = data


async def attitude_listener(drone):
    async for data in drone.telemetry.attitude_euler():
        latest["att"] = data


async def battery_listener(drone):
    async for data in drone.telemetry.battery():
        latest["bat"] = data


# ============================================================
# EXIF helpers
# ============================================================

def deg_to_dms_rational(deg_float):
    deg = int(deg_float)

    minutes_float = (deg_float - deg) * 60
    minutes = int(minutes_float)

    seconds = (minutes_float - minutes) * 60

    return (
        (deg, 1),
        (minutes, 1),
        (int(round(seconds * 100)), 100),
    )


def altitude_to_rational(altitude_m):
    return (
        int(round(abs(float(altitude_m)) * 100)),
        100,
    )


def create_exif(
    timestamp,
    gps=None,
    pos=None,
    att=None,
    bat=None,
):
    """
    Create EXIF containing GPS information plus a human-readable
    telemetry record in ImageDescription.
    """

    timestamp_string = timestamp.strftime(
        "%Y:%m:%d %H:%M:%S"
    )

    description = []

    # --------------------------------------------------------
    # GPS
    # --------------------------------------------------------

    if gps is not None:
        description.append(
            f"SATS={gps.num_satellites}"
        )

        description.append(
            f"FIX={gps.fix_type}"
        )

    # --------------------------------------------------------
    # Position
    # --------------------------------------------------------

    if pos is not None:
        description.append(
            f"LAT={pos.latitude_deg:.7f}"
        )

        description.append(
            f"LON={pos.longitude_deg:.7f}"
        )

        description.append(
            f"REL_ALT={pos.relative_altitude_m:.3f}m"
        )

        description.append(
            f"ABS_ALT={pos.absolute_altitude_m:.3f}m"
        )

    # --------------------------------------------------------
    # Attitude
    # --------------------------------------------------------

    if att is not None:
        description.append(
            f"YAW={att.yaw_deg:.3f}"
        )

        description.append(
            f"PITCH={att.pitch_deg:.3f}"
        )

        description.append(
            f"ROLL={att.roll_deg:.3f}"
        )

    # --------------------------------------------------------
    # Battery
    # --------------------------------------------------------

    if bat is not None:
        description.append(
            f"BAT_V={bat.voltage_v:.3f}"
        )

        description.append(
            f"BAT_A={bat.current_battery_a:.3f}"
        )

    description_string = " | ".join(description)

    if not description_string:
        description_string = "NO_TELEMETRY"

    # --------------------------------------------------------
    # EXIF dictionary
    # --------------------------------------------------------

    exif_data = {
        "0th": {
            piexif.ImageIFD.Artist:
                "Jetson ML Capture",

            piexif.ImageIFD.Make:
                "Raspberry Pi",

            piexif.ImageIFD.Model:
                "HQ Camera IMX477",

            piexif.ImageIFD.Software:
                "Jetson + OpenCV + MAVSDK",

            piexif.ImageIFD.ImageDescription:
                description_string,
        },

        "Exif": {
            piexif.ExifIFD.DateTimeOriginal:
                timestamp_string,

            piexif.ExifIFD.DateTimeDigitized:
                timestamp_string,
        },

        "GPS": {},
    }

    # --------------------------------------------------------
    # GPS EXIF
    # --------------------------------------------------------

    if pos is not None:

        lat = pos.latitude_deg
        lon = pos.longitude_deg
        alt = pos.absolute_altitude_m

        exif_data["GPS"] = {

            piexif.GPSIFD.GPSLatitudeRef:
                "N" if lat >= 0 else "S",

            piexif.GPSIFD.GPSLatitude:
                deg_to_dms_rational(abs(lat)),

            piexif.GPSIFD.GPSLongitudeRef:
                "E" if lon >= 0 else "W",

            piexif.GPSIFD.GPSLongitude:
                deg_to_dms_rational(abs(lon)),

            piexif.GPSIFD.GPSAltitudeRef:
                0 if alt >= 0 else 1,

            piexif.GPSIFD.GPSAltitude:
                altitude_to_rational(alt),
        }

    return piexif.dump(exif_data)


# ============================================================
# Main
# ============================================================

async def main():

    shutdown_event = asyncio.Event()

    loop = asyncio.get_event_loop()

    loop.add_signal_handler(
        signal.SIGINT,
        shutdown_event.set
    )

    # ========================================================
    # Connect to Cube Orange+
    # ========================================================

    print("Connecting to Cube Orange+...")

    drone = System()

    await drone.connect(
        system_address=SERIAL_ADDRESS
    )

    async for state in drone.core.connection_state():

        if state.is_connected:

            print("Cube connected.")

            break

    # ========================================================
    # Request telemetry rates
    # ========================================================

    print("Requesting telemetry rates...")

    await drone.telemetry.set_rate_gps_info(1)
    await drone.telemetry.set_rate_position(5)
    await drone.telemetry.set_rate_attitude_euler(10)
    await drone.telemetry.set_rate_battery(1)

    # ========================================================
    # Start telemetry listeners
    # ========================================================

    asyncio.create_task(
        gps_listener(drone)
    )

    asyncio.create_task(
        position_listener(drone)
    )

    asyncio.create_task(
        attitude_listener(drone)
    )

    asyncio.create_task(
        battery_listener(drone)
    )

    # Give telemetry a moment to populate
    await asyncio.sleep(2)

    # ========================================================
    # Open camera
    # ========================================================

    print("Opening camera...")

    pipeline = gstreamer_pipeline()

    cam = cv2.VideoCapture(
        pipeline,
        cv2.CAP_GSTREAMER
    )

    if not cam.isOpened():

        print("ERROR: Camera failed to open.")

        return

    print("Camera opened.")

    # ========================================================
    # Flush old camera frames
    # ========================================================

    print("Flushing camera buffer...")

    for _ in range(20):
        cam.grab()

    print("Camera ready.")
    print()

    print("==========================================")
    print("ML IMAGE CAPTURE")
    print("==========================================")
    print(f"Resolution : {CAMERA_WIDTH} x {CAMERA_HEIGHT}")
    print(f"Interval   : {PHOTO_INTERVAL} seconds")
    print(f"JPEG       : {JPEG_QUALITY}")
    print(f"Output     : {OUTPUT_DIR}")
    print("Telemetry  : Cube Orange+ / MAVLink")
    print("Press CTRL+C to stop.")
    print("==========================================")
    print()

    image_number = 0

    next_capture = time.monotonic()

    try:

        while not shutdown_event.is_set():

            # ------------------------------------------------
            # Wait for scheduled capture
            # ------------------------------------------------

            now = time.monotonic()

            wait_time = next_capture - now

            if wait_time > 0:

                try:
                    await asyncio.wait_for(
                        shutdown_event.wait(),
                        timeout=wait_time
                    )
                except asyncio.TimeoutError:
                    pass

            if shutdown_event.is_set():
                break

            # ------------------------------------------------
            # Read newest camera frame
            # ------------------------------------------------

            ret, frame = cam.read()

            if not ret:

                print(
                    "ERROR: Failed to read camera frame"
                )

                next_capture += PHOTO_INTERVAL

                continue

            # ------------------------------------------------
            # Timestamp capture
            # ------------------------------------------------

            capture_time = datetime.datetime.now()

            # ------------------------------------------------
            # Take telemetry snapshot at capture time
            # ------------------------------------------------

            gps = latest["gps"]
            pos = latest["pos"]
            att = latest["att"]
            bat = latest["bat"]

            # ------------------------------------------------
            # Create filename
            # ------------------------------------------------

            filename = os.path.join(
                OUTPUT_DIR,
                capture_time.strftime(
                    "%Y%m%d_%H%M%S_%f"
                )[:-3] + ".jpg"
            )

            # ------------------------------------------------
            # Create EXIF
            # ------------------------------------------------

            exif = create_exif(
                capture_time,
                gps=gps,
                pos=pos,
                att=att,
                bat=bat,
            )

            # ------------------------------------------------
            # Save image
            # ------------------------------------------------

            success = cv2.imwrite(
                filename,
                frame,
                [
                    cv2.IMWRITE_JPEG_QUALITY,
                    JPEG_QUALITY
                ]
            )

            if success:

                # Open the saved JPEG and insert EXIF.
                #
                # OpenCV's imwrite does not preserve/write
                # arbitrary EXIF data, so piexif is used here.
                with open(filename, "rb") as f:
                    jpeg_data = f.read()

                jpeg_data = piexif.insert(
                    exif,
                    jpeg_data
                )

                with open(filename, "wb") as f:
                    f.write(jpeg_data)

                image_number += 1

                # ------------------------------------------------
                # Console telemetry summary
                # ------------------------------------------------

                print()
                print(
                    f"[{capture_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}]"
                )

                print(
                    f"  Image: #{image_number}"
                )

                if pos is not None:

                    print(
                        f"  GPS: "
                        f"{pos.latitude_deg:.7f}, "
                        f"{pos.longitude_deg:.7f}"
                    )

                    print(
                        f"  Altitude: "
                        f"{pos.absolute_altitude_m:.2f} m"
                    )

                else:

                    print("  GPS: unavailable")

                if att is not None:

                    print(
                        f"  Attitude: "
                        f"yaw={att.yaw_deg:.2f} "
                        f"pitch={att.pitch_deg:.2f} "
                        f"roll={att.roll_deg:.2f}"
                    )

                else:

                    print("  Attitude: unavailable")

                if gps is not None:

                    print(
                        f"  GPS: "
                        f"{gps.num_satellites} satellites, "
                        f"fix={gps.fix_type}"
                    )

                print(
                    f"  Saved: {filename}"
                )

            else:

                print(
                    "ERROR: Failed to save image"
                )

            # ------------------------------------------------
            # Schedule next capture
            # ------------------------------------------------

            next_capture += PHOTO_INTERVAL

            # Don't rapidly catch up if something took too long
            if next_capture < time.monotonic():

                next_capture = (
                    time.monotonic()
                    + PHOTO_INTERVAL
                )

    finally:

        cam.release()

        print()
        print(
            f"Total images captured: "
            f"{image_number}"
        )

        print("Camera released.")
        print("Shutdown complete.")


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":

    try:

        asyncio.run(main())

    except KeyboardInterrupt:

        print("Interrupted")

import asyncio
import cv2
import os
import time
import datetime
import signal
from PIL import Image
import piexif
from mavsdk import System

# Constants
OUTPUT_DIR = "images"
MIN_SATELLITES = 6
PHOTO_INTERVAL = 3
CAMERA_SENSOR_ID = 0
FLIP_METHOD = 2

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Shared telemetry store
latest = {
    "gps": None,
    "pos": None,
    "att": None,
    "bat": None
}

# ---------------- Camera ----------------
def gstreamer_pipeline(sensor_id=0, width=1920, height=1080, framerate=10, flip_method=2):
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} wbmode=1 ! "
        f"video/x-raw(memory:NVMM), width=(int){width}, height=(int){height}, framerate=(fraction){framerate}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw, format=(string)BGRx ! "
        f"videoconvert ! video/x-raw, format=(string)BGR ! appsink"
    )

# ---------------- EXIF ----------------
def deg_to_dms_rational(deg_float):
    deg = int(deg_float)
    min_float = (deg_float - deg) * 60
    min_ = int(min_float)
    sec = int((min_float - min_) * 60 * 100)
    return ((deg, 1), (min_, 1), (sec, 100))

def create_exif(lat, lon, timestamp, description):
    gps_ifd = {
        piexif.GPSIFD.GPSLatitudeRef: 'N' if lat >= 0 else 'S',
        piexif.GPSIFD.GPSLatitude: deg_to_dms_rational(abs(lat)),
        piexif.GPSIFD.GPSLongitudeRef: 'E' if lon >= 0 else 'W',
        piexif.GPSIFD.GPSLongitude: deg_to_dms_rational(abs(lon)),
    }
    exif_datetime = timestamp.strftime("%Y:%m:%d %H:%M:%S")

    return piexif.dump({
        "0th": {
            piexif.ImageIFD.Artist: u"JetsonCam",
            piexif.ImageIFD.Make: u"Jetson",
            piexif.ImageIFD.Model: u"IMX219",
            piexif.ImageIFD.Software: u"OpenCV + piexif",
            piexif.ImageIFD.ImageDescription: description,
        },
        "Exif": {
            piexif.ExifIFD.DateTimeOriginal: exif_datetime,
            piexif.ExifIFD.DateTimeDigitized: exif_datetime,
        },
        "GPS": gps_ifd,
    })

def create_basic_exif(timestamp, description):
    exif_datetime = timestamp.strftime("%Y:%m:%d %H:%M:%S")
    return piexif.dump({
        "0th": {
            piexif.ImageIFD.ImageDescription: description,
        },
        "Exif": {
            piexif.ExifIFD.DateTimeOriginal: exif_datetime,
        },
    })

# ---------------- Telemetry Listeners ----------------
async def gps_listener(drone):
    async for data in drone.telemetry.gps_info():
        latest["gps"] = data

async def pos_listener(drone):
    async for data in drone.telemetry.position():
        latest["pos"] = data

async def att_listener(drone):
    async for data in drone.telemetry.attitude_euler():
        latest["att"] = data

async def bat_listener(drone):
    async for data in drone.telemetry.battery():
        latest["bat"] = data

# ---------------- Main ----------------
async def main():
    shutdown_event = asyncio.Event()
    loop = asyncio.get_event_loop()
    loop.add_signal_handler(signal.SIGINT, shutdown_event.set)

    drone = System()
    print("Connecting to PX4...")
    await drone.connect(system_address="serial:///dev/ttyACM0:57600")

    async for state in drone.core.connection_state():
        if state.is_connected:
            print("Connected.")
            break

    await asyncio.sleep(2)

    print("Requesting telemetry rates...")
    await drone.telemetry.set_rate_position(10)
    await drone.telemetry.set_rate_battery(1)
    await drone.telemetry.set_rate_attitude_euler(10)
    await drone.telemetry.set_rate_imu(10)

    # Start telemetry listeners
    asyncio.create_task(gps_listener(drone))
    asyncio.create_task(pos_listener(drone))
    asyncio.create_task(att_listener(drone))
    asyncio.create_task(bat_listener(drone))

    # Camera
    cam = cv2.VideoCapture(
        gstreamer_pipeline(sensor_id=CAMERA_SENSOR_ID, flip_method=FLIP_METHOD),
        cv2.CAP_GSTREAMER
    )

    if not cam.isOpened():
        print("Camera failed.")
        return

    print("Starting capture loop...")

    try:
        while not shutdown_event.is_set():
            timestamp = datetime.datetime.now()

            gps = latest["gps"]
            pos = latest["pos"]
            att = latest["att"]
            bat = latest["bat"]

            yaw = pitch = roll = None
            error_msg = None

            if att is not None:
                yaw = att.yaw_deg
                pitch = att.pitch_deg
                roll = att.roll_deg
            else:
                error_msg = "No attitude"

            # Flush buffer
            for _ in range(5):
                cam.read()
                time.sleep(0.02)

            ret, frame = cam.read()
            if not ret:
                print(f"[{timestamp}] Frame failed")
                continue

            filename = os.path.join(
                OUTPUT_DIR,
                timestamp.strftime("%Y-%m-%d_%H-%M-%S.jpeg")
            )

            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            # Build description
            desc = []

            if gps is not None:
                desc.append(f"SATS: {gps.num_satellites}")

            if yaw is not None:
                desc.append(f"YAW: {yaw:.2f}")
            if pitch is not None:
                desc.append(f"PITCH: {pitch:.2f}")
            if roll is not None:
                desc.append(f"ROLL: {roll:.2f}")

            if bat is not None:
                desc.append(f"BAT: {bat.voltage_v:.2f}V")

            if error_msg:
                desc.append(f"ERR: {error_msg}")

            description = " | ".join(desc) if desc else "No telemetry"

            # Save image
            if gps and pos and gps.num_satellites >= MIN_SATELLITES:
                exif = create_exif(
                    pos.latitude_deg,
                    pos.longitude_deg,
                    timestamp,
                    description
                )
                img.save(filename, "jpeg", exif=exif)
                print(f"[{timestamp}] Saved WITH GPS")
            else:
                exif = create_basic_exif(timestamp, description)
                img.save(filename, "jpeg", exif=exif)
                print(f"[{timestamp}] Saved WITHOUT GPS")

            await asyncio.sleep(PHOTO_INTERVAL)

    finally:
        cam.release()
        print("Shutdown complete.")

# ---------------- Entry ----------------
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Interrupted")

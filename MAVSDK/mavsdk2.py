import asyncio
from mavsdk import System

async def run():
    drone = System()
    print("Connecting to Ardupilot...")
    await drone.connect(system_address="serial:///dev/ttyACM0:921600")

    print("Waiting for connection...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print("Connected!")
            break

    # Give ArduPilot time to initialize MAVLink streams
    await asyncio.sleep(2)

    print("Requesting telemetry rates...")
    try:
        await drone.telemetry.set_rate_battery(1)
        await drone.telemetry.set_rate_attitude_euler(5)  # helps "wake up" stream
        await drone.telemetry.set_rate_imu(5)
    except Exception as e:
        print(f"Rate request failed: {e}")

    print("Waiting for battery data...")

    # Timeout protection so you know if it's failing
    got_data = False

    async for battery in drone.telemetry.battery():
        print(f"Battery Voltage: {battery.voltage_v:.2f} V")
        got_data = True

        # Print once and exit (remove break if you want continuous)
        break

    if not got_data:
        print("No battery data received.")

if __name__ == "__main__":
    asyncio.run(run())

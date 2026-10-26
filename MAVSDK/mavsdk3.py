import asyncio
from mavsdk import System

latest_battery = None
latest_attitude = None

async def battery_listener(drone):
    global latest_battery
    async for battery in drone.telemetry.battery():
        latest_battery = battery  # always keep latest

async def attitude_listener(drone):
    global latest_attitude
    async for attitude in drone.telemetry.attitude_euler():
        latest_attitude = attitude  # always keep latest

async def printer():
    while True:
        if latest_battery and latest_attitude:
            print(
                f"\r[BATTERY] {latest_battery.voltage_v:.2f} V | "
                f"[ATTITUDE] R:{latest_attitude.roll_deg:.1f} "
                f"P:{latest_attitude.pitch_deg:.1f} "
                f"Y:{latest_attitude.yaw_deg:.1f}",
                end=""
            )
        await asyncio.sleep(0.1)  # control output rate ONLY here

async def run():
    drone = System()
    print("Connecting to Ardupilot...")
    await drone.connect(system_address="serial:///dev/ttyACM0:57600")

    print("Waiting for connection...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print("Connected!")
            break

    await asyncio.sleep(2)

    print("Requesting telemetry rates...")
    await drone.telemetry.set_rate_battery(1)
    await drone.telemetry.set_rate_attitude_euler(10)
    await drone.telemetry.set_rate_imu(10)

    print("Starting telemetry...\n")

    await asyncio.gather(
        battery_listener(drone),
        attitude_listener(drone),
        printer(),
    )

if __name__ == "__main__":
    asyncio.run(run())

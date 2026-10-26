import asyncio
from mavsdk import System

async def print_health(drone):
    async for health in drone.telemetry.health():
        print(f"[HEALTH] global_ok={health.is_global_position_ok} "
              f"home_ok={health.is_home_position_ok}")

async def print_gps_info(drone):
    async for gps in drone.telemetry.gps_info():
        print(f"[GPS_INFO] sats={gps.num_satellites} fix={gps.fix_type}")

async def print_position(drone):
    async for pos in drone.telemetry.position():
        print(f"[POSITION] lat={pos.latitude_deg:.6f}, "
              f"lon={pos.longitude_deg:.6f}, "
              f"alt={pos.absolute_altitude_m:.2f}")

async def main():
    drone = System()

    print("Connecting via serial...")
    await drone.connect(system_address="serial:///dev/ttyACM0:57600")

    print("Waiting for connection...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print("✅ Connected")
            break

    # Give autopilot a moment
    await asyncio.sleep(2)

    print("Requesting telemetry rates...")
    await drone.telemetry.set_rate_gps_info(1)
    await drone.telemetry.set_rate_position(1)

    print("Starting telemetry streams...\n")

    await asyncio.gather(
        print_health(drone),
        print_gps_info(drone),
        print_position(drone),
    )

if __name__ == "__main__":
    asyncio.run(main())
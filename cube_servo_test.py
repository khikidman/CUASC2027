#!/usr/bin/env python3
"""Interactive MAVLink servo tester for Cube Orange+ MAIN OUT 1 (ArduPilot)."""

import argparse
import time

from pymavlink import mavutil
from serial.tools import list_ports

SERVO_CHANNEL = 1      # Cube Orange+ MAIN OUT 1
MIN_PWM = 500         # Safe initial calibration range
MAX_PWM = 2500
CENTER_PWM = 1500


def choose_port(explicit_port):
    if explicit_port:
        return explicit_port
    ports = list(list_ports.comports())
    usb = [p for p in ports if (
        "usbmodem" in p.device.lower() or
        "ttyacm" in p.device.lower() or
        "com" == p.device[:3].lower()
    )]
    if len(usb) == 1:
        print(f"Detected serial device: {usb[0].device}")
        return usb[0].device
    print("Available serial ports:")
    for p in ports:
        print(f"  {p.device} — {p.description}")
    raise SystemExit("Specify the Cube port with --port (e.g. --port /dev/cu.usbmodem101).")


def set_pwm(master, pwm):
    pwm = round(pwm)
    if not MIN_PWM <= pwm <= MAX_PWM:
        print(f"Rejected: PWM must be {MIN_PWM}–{MAX_PWM} µs")
        return

    cmd = mavutil.mavlink.MAV_CMD_DO_SET_SERVO
    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        cmd,
        0,
        SERVO_CHANNEL,
        pwm,
        0, 0, 0, 0, 0,
    )
    print(f"Sent MAIN OUT {SERVO_CHANNEL}: {pwm} µs")
    # ACK confirms command handling, not physical movement.
    deadline = time.monotonic() + 1.5
    while time.monotonic() < deadline:
        ack = master.recv_match(type="COMMAND_ACK", blocking=True, timeout=0.25)
        if ack and ack.command == cmd:
            results = mavutil.mavlink.enums["MAV_RESULT"]
            result = results.get(ack.result)
            print(f"  ArduPilot: {result.name if result else ack.result}")
            return
    print("  No acknowledgment (check connection/output settings)")


def angle_to_pwm(angle):
    return MIN_PWM + (MAX_PWM - MIN_PWM) * angle / 270.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", help="Cube USB serial port")
    parser.add_argument("--baud", type=int, default=115200,
                        help="Baud rate for UART; USB ignores this setting")
    args = parser.parse_args()
    port = choose_port(args.port)
    print(f"Connecting to {port}...")
    master = mavutil.mavlink_connection(port, baud=args.baud)
    try:
        heartbeat = master.wait_heartbeat(timeout=10)
        if heartbeat is None:
            raise RuntimeError("No Cube heartbeat. Check port and close QGroundControl.")
        print(f"Connected: system {master.target_system}, component {master.target_component}")
        print("Commands: c=center, a ANGLE (0–270), p PWM (1000–2000),")
        print("          sweep=gentle sweep (1400/1600), q=quit")
        print("No PWM command has been sent yet.")

        while True:
            try:
                parts = input("servo> ").strip().lower().split()
            except EOFError:
                break
            if not parts:
                continue
            try:
                if parts[0] in ("q", "quit", "exit"):
                    break
                elif parts[0] in ("c", "center"):
                    set_pwm(master, CENTER_PWM)
                elif parts[0] in ("a", "angle") and len(parts) == 2:
                    angle = float(parts[1])
                    if not 0 <= angle <= 180:
                        print("Angle must be between 0 and 180 degrees")
                    else:
                        set_pwm(master, angle_to_pwm(angle))
                elif parts[0] in ("p", "pwm") and len(parts) == 2:
                    set_pwm(master, float(parts[1]))
                elif parts[0] == "sweep":
                    for pwm in (CENTER_PWM, 1400, CENTER_PWM, 1600, CENTER_PWM):
                        set_pwm(master, pwm)
                        time.sleep(1)
                else:
                    print("Use: c | a 90 | p 1500 | sweep | q")
            except ValueError:
                print("Enter a valid number, e.g. a 90 or p 1500")
    except KeyboardInterrupt:
        print("\nStopped by user")
    finally:
        master.close()
        print("Connection closed. Servo may hold its last position.")


if __name__ == "__main__":
    main()

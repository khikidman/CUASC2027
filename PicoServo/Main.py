from machine import Pin, PWM
import time

# Initialize PWM for servo control
servo = PWM(Pin(16))
servo.freq(50)  # Set frequency to 50 Hz

# Function to set servo angle

def set_servo_angle(angle):
    # Clamp angle between 0 and 180
    angle = max(0, min(180, angle))
    
    # Map 0-180 degrees to approximate 50Hz 16-bit duty cycle range (adjust min/max as needed for your specific servo)
    # 0 deg  -> ~1.0 ms pulse -> duty_u16 ≈ 3277
    # 180 deg -> ~2.0 ms pulse -> duty_u16 ≈ 6554
    # Or standard range: 1638 (0.5ms) to 8191 (2.5ms)
    min_duty = 1638  # ~0.5ms
    max_duty = 8191  # ~2.5ms
    
    duty = int(min_duty + (angle / 180) * (max_duty - min_duty))
    servo.duty_u16(duty)
    return duty

# Example usage
set_servo_angle(90)  # Set servo to 90 degrees
set_servo_angle(0)   # Set servo to 0 degrees
set_servo_angle(180) # Set servo to 180 degrees

try:
    set_servo_angle(180)  # Set servo to 180 degrees
    time.sleep(1)
    set_servo_angle(0)  # Set servo to 0 degrees
    time.sleep(1)
except Exception as e:
    print("Error setting servo angle:", e)
    servo.duty_u16(0)  # Turn off servo in case of error


from machine import PWM, Pin

pin = Pin("LED", Pin.OUT)
pin.value(1)  # Turn on the LED
pwm = PWM(Pin(15, Pin.OUT), freq=50)

def set_angle(angle):
    pulse_us = 500 + ((angle/270) * 2000) # divided by 10^3
    pwm.duty_ns(round(pulse_us * 1000))

set_angle(0)

while True:
    user_input = input("Angle: ")
    if user_input.lower() == "stop":
        pwm.deinit()
        break
    try:
        set_angle(float(user_input))
    except ValueError as e:
        print("Invalid input:", e)

pin.value(0)  # Turn off the LED
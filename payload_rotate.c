#include <stdio.h>
#include "pico/stdlib.h"
#include "hardware/pwm.h"

#define SERVO_PIN 9  // Change to your GPIO

// Convert angle (0–180) to pulse width in microseconds
uint16_t angle_to_us(float angle) {
    return 1000 + (angle / 180.0f) * 1000; // 1000–2000 us
}

void open_claw() {
    uint16_t pulse = angle_to_us(0);
    pwm_set_gpio_level(SERVO_PIN, pulse);

    printf("Angle: %.1f°, Pulse: %d us\n", 0, pulse);
}

void close_claw() {
    uint16_t pulse = angle_to_us(180);
    pwm_set_gpio_level(SERVO_PIN, pulse);

    printf("Angle: %.1f°, Pulse: %d us\n", 180, pulse);
}

int main() {
    stdio_init_all();
    printf("Hello world");

    // Set GPIO to PWM
    gpio_set_function(SERVO_PIN, GPIO_FUNC_PWM);

    // Get PWM slice
    uint slice_num = pwm_gpio_to_slice_num(SERVO_PIN);

    // Set PWM frequency to 50 Hz
    // Pico runs at 125 MHz default
    // We want: 50 Hz → 20 ms period
    // Use divider + wrap
    pwm_set_clkdiv(slice_num, 125.0f); // 125 MHz / 125 = 1 MHz (1 tick = 1 us)
    pwm_set_wrap(slice_num, 20000);    // 20,000 ticks → 20 ms

    pwm_set_enabled(slice_num, true);



    while (1) {
        open_claw();
        sleep_ms(1000);
        close_claw();
        sleep_ms(1000);
    }
}

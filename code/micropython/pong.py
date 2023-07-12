from fpioa_manager import fm, board_info
from machine import I2C
import sensor, image, lcd
from time import sleep

lcd.init()

# game parameters
ball_pos = [120, 160]
ball_dir = [1, 1]
ball_radius = 10

paddle_pos = [100, 200]
paddle_width = 40
paddle_speed = 5

# game loop
while True:
    # clear the display
    lcd.clear()

    # move the ball
    ball_pos[0] += ball_dir[0]
    ball_pos[1] += ball_dir[1]

    # check for collision with the wall
    if ball_pos[0] < 0 or ball_pos[0] > lcd.width():
        ball_dir[0] = -ball_dir[0]
    if ball_pos[1] < 0:
        ball_dir[1] = -ball_dir[1]

    # check for collision with the paddle
    if ball_pos[1] > paddle_pos[1] and paddle_pos[0] < ball_pos[0] < paddle_pos[0] + paddle_width:
        ball_dir[1] = -ball_dir[1]

    # draw the ball
    lcd.circle(ball_pos[0], ball_pos[1], ball_radius, lcd.RED, lcd.RED)

    # move the paddle based on button input
    # assume button_pin is the GPIO pin the button is connected to
    button_pin = board_info.D[0]
    fm.register(button_pin, fm.fpioa.GPIO0)
    button = GPIO(GPIO.GPIO0, GPIO.PULL_UP)
    if button.value() == 0:
        paddle_pos[0] -= paddle_speed
    else:
        paddle_pos[0] += paddle_speed

    # draw the paddle
    lcd.rect(paddle_pos[0], paddle_pos[1], paddle_pos[0] + paddle_width, paddle_pos[1] + 10, lcd.GREEN, lcd.GREEN)

    # delay before the next frame
    sleep(0.02)

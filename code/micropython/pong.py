import sensor, lcd, image, time, urandom
from fpioa_manager import fm
from board import board_info
from maix import GPIO

from maix import KPU
kpu = KPU()
kpu.load("/flash/cdn.kmodel")

sensor.reset()                      # 复位并初始化摄像头
sensor.set_pixformat(sensor.RGB565) # 设置摄像头输出格式为 RGB565（也可以是GRAYSCALE）
sensor.set_framesize(sensor.QVGA)   # 设置摄像头输出大小为 QVGA (320x240)
sensor.skip_frames(time = 2000)     # 跳过2000帧
clock = time.clock()                # 创建一个clock对象，用来计算帧率

lcd.init()                          # Init lcd display
lcd.clear(lcd.RED)                  # Clear lcd screen.

fm.register(board_info.BOOT_KEY, fm.fpioa.GPIO3, force=True)
key_input = GPIO(GPIO.GPIO3, GPIO.IN)
# Game parameters

SCALE = 3

WIDTH, HEIGHT = 600//SCALE, 600//SCALE
BALL_SIZE = 20 //SCALE
PADDLE_W, PADDLE_H, PADDLE_SPEED = 40//SCALE, 250//SCALE, 10//SCALE
FONT_SIZE = 32//SCALE
BALL_SPEED = 1


ENV_X, ENV_Y = 50, 15

R,P,N = 1,-5,0

PLAYER_2 = False

class Paddle:
    def __init__(self, paddle_w, paddle_h, speed=10, right=False):
        """
        Paddle class
        Args:
            paddle_w: paddle width, in pixels
            paddle_h: paddle height, in pixels
            screen_w: screen width, in pixels
            screen_h: screen height, in pixels
            speed: paddle speed, in pixels
        """
        self.x = 0
        if right:
            self.x = WIDTH
        self.y = HEIGHT // 2
        self.w = paddle_w
        self.h = paddle_h
        self.screen_h = HEIGHT
        self.screen_w = WIDTH
        self.speed = speed


    def move(self, direction):
        if direction == 1 and self.y + self.h//2 + self.speed < self.screen_h:
            self.y += self.speed
        elif direction == -1 and self.y - self.h//2 > self.speed:
            self.y -= self.speed

class Ball:
    def __init__(self, dx=1, dy=1, ball_size=40, ball_speed=3):
        """
        Ball class
        Args:
            dx: velocity in x direction
            dy: velocity in y direction
            screen_w: screen width, in pixels
            screen_h: screen height, in pixels
            ball_size: ball size, in pixels
        """
        self.x = WIDTH// 2
        self.y = HEIGHT // 2
        self.speed = ball_speed
        self.dx = dx
        self.dy = dy
        self.screen_h = HEIGHT
        self.screen_w = WIDTH
        self.ball_size = ball_size

    def move(self, player_2=False):
        bounced = False
        self.x += self.dx * self.speed
        self.y += self.dy * self.speed

        if self.y - self.ball_size < 0 or self.y > self.screen_h - self.ball_size:
            self.dy *= -1
            bounced = True

        if not player_2 and self.x > self.screen_w - self.ball_size:  # Adding bouncing for the third border
            self.dx *= -1
            bounced = True
        return bounced

    def check_collision(self, paddle):
        return self.check_collision_left(paddle)

    def check_collision_left(self, paddle):
        if self.dx < 0 and self.x - self.ball_size < paddle.x + paddle.w and paddle.y - paddle.h//2 < self.y < paddle.y + paddle.h//2:
            self.dx *= -1
            return True
        return False

    def check_collision_right(self, paddle):
        if self.dx > 0 and self.x + self.ball_size > paddle.x - paddle.w and paddle.y - paddle.h//2 < self.y < paddle.y + paddle.h//2:
            self.dx *= -1
            return True
        return False

def draw_ball(ball):
    img.draw_circle(ENV_X+ball.x, ENV_Y+ball.y, ball.ball_size,(255,255,255), 1, fill=True)

def draw_paddle(paddle):
    img.draw_rectangle(ENV_X+paddle.x//2,ENV_Y+paddle.y-paddle.h//2,paddle.w,paddle.h, color=(255,255,255), fill=True)

def draw_paddle_right(paddle):
    img.draw_rectangle(ENV_X+paddle.x-paddle.w ,ENV_Y+paddle.y-paddle.h//2,paddle.w,paddle.h, color=(255,255,255), fill=True)

def draw_env():
    img.draw_rectangle(0,0,320,240,(0,0,0), fill=True)
    thickness = 1
    img.draw_rectangle(ENV_X,ENV_Y,WIDTH,HEIGHT, (0,0,0), thickness, fill=True)
    img.draw_rectangle(ENV_X,ENV_Y,WIDTH,HEIGHT, (255,255,255),thickness, fill=False)

    s = "NEUROPONG"
    for i in range(len(s)):
        img.draw_string(10, i*22+15, s[i], color = (255, 255, 255), scale = 2, mono_space = False,
                            char_rotation = 0, char_hmirror = False, char_vflip = False,
                            string_rotation = 0, string_hmirror = False, string_vflip = True)

def draw_score(score):
    img.draw_string(180, 17, "score : "+str(score), color = (255, 255, 255), scale = 1, mono_space = False,
                            char_rotation = 0, char_hmirror = False, char_vflip = False,
                            string_rotation = 0, string_hmirror = False, string_vflip = True)


img = sensor.snapshot()

def game_over():
    img.draw_rectangle(0,0,320,240, (0,0,0), fill=True)
    img.draw_string(100, 100, "GAME OVER", color = (255, 0, 0), scale = 2, mono_space = False,
                        char_rotation = 0, char_hmirror = False, char_vflip = False,
                        string_rotation = 0, string_hmirror = False, string_vflip = False)

    img.draw_rectangle(90,90,120,43, (255,255,255), 3, fill=False)




def get_action(paddle, ball):
    img = sensor.snapshot()
    img_mnist1=img.to_grayscale(1)
    img_mnist2=img_mnist1.resize(1,2)

    y_p = paddle.y
    y_b = ball.y

    # add noise
    n = 6
    noise = urandom.randint(-n,n)

    img_mnist2.set_pixel(0,0,y_p)
    img_mnist2.set_pixel(0,1,y_b+noise)

    img_mnist2.pix_to_ai()
    kpu.run(img_mnist2)
    feature_map = kpu.get_outputs()
    if feature_map[0]>feature_map[1]:
       return 1
    else:
       return -1

def render(paddle, ball, paddle_right=0):
    draw_env()
    draw_paddle(paddle)
    draw_ball(ball)
    if paddle_right!=0:
       draw_paddle_right(paddle_right)

def pseudo_ai(paddle, ball):
   DEBUG=False
   if DEBUG:
      if paddle.y < ball.y:
         return 1
      else:
         return -1
   else:
      return get_action(paddle, ball)

def game_loop(FPS):
    # Game Initialization

    running = True

    # RL Agent
    num_inputs = 5
    num_actions = 2

    iteration = 0
    episode = 0
    score = 0

    while running:

        done = False
        paddle = Paddle(PADDLE_W, PADDLE_H, PADDLE_SPEED)
        paddle_right = Paddle(PADDLE_W, PADDLE_H, PADDLE_SPEED, right=True)
        dx = 12
        dy = 4

        ball = Ball(dx,dy, BALL_SIZE, ball_speed=BALL_SPEED)
        score = 0
        game_over_flag = False

        state = [paddle.y, ball.x, ball.y, ball.dx, ball.dy]
        next_state = state

        while not game_over_flag:
            img = sensor.snapshot()

            action = pseudo_ai(paddle, ball)
            paddle.move(action)

            if PLAYER_2:
                if key_input.value()==0:
                    paddle_right.move(-1)
                else:
                    paddle_right.move(1)

            bounced = ball.move(player_2=PLAYER_2)
            done = ball.x < 0 or ball.x > WIDTH

            if PLAYER_2:
                collided = ball.check_collision_left(paddle)
                collided = ball.check_collision_right(paddle_right)
            else:
                collided = ball.check_collision(paddle)

            reward = N
            if collided:
                reward = R
            elif done:
                reward = P
            else:
                reward = N

            if PLAYER_2:
                render(paddle, ball, paddle_right)
            else:
                render(paddle, ball)

            # update screen and scores
            score += (1 if collided else 0)
            if done or score > 100:  # episode is over if the score is greater than 100
                episode += 1
                game_over()
                lcd.display(img)
                time.sleep(4)
                game_over_flag = True

            draw_score(score)
            lcd.display(img)                # Display image on lcd.
            print(clock.fps()) # 打印帧率

game_loop(100)

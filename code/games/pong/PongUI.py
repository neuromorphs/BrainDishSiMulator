import pygame
import time
import cv2
import numpy as np



# Pygame Initialization
pygame.init()

# Game parameters
WIDTH, HEIGHT = 600, 600
BALL_SIZE = 40
PADDLE_W, PADDLE_H = 40, 250
FPS = 100
FONT_SIZE = 32

# Player type
PLAYER = "RL-AGENT"  # Choose between "AI" and "Human"

# Colors
WHITE = (255, 255, 255)
RED = (255, 0, 0)

# Game Objects
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()
font = pygame.font.Font(None, FONT_SIZE)


def draw_env_status():
    downscaled = cv2.resize(pygame.surfarray.array3d(screen), (40, 40))
    return downscaled


def draw_header(score, generation):
    score_text = font.render(f"Generation {generation} - Score: {score}", True, WHITE)
    screen.blit(score_text, (WIDTH - score_text.get_width() - 10, 10))


def game_over():
    game_over_text = font.render("Fail", True, RED)
    screen.blit(game_over_text, (WIDTH // 2 - game_over_text.get_width() // 2, HEIGHT // 2))


def pseudo_ai(paddle, ball):
    if paddle.y + PADDLE_H / 2 < ball.y:
        return -1
    else:
        return 1

def game_loop():
    paddle = Paddle()
    ball = Ball()
    score = 0
    running = True
    generation = 0

    # RL Agent
    agent = RLAgent()
    seed = 0
    np.random.seed(seed)

    while running:
        done = False
        paddle = Paddle()
        angle = np.random.choice([45, 135, 225, 315])
        dx = np.cos(angle)
        dy = np.sin(angle)
        ball = Ball(dx, dy)
        score = 0
        game_over_flag = False
        while not game_over_flag:
            screen.fill((0, 0, 0))

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            if PLAYER == "HUMAN":
                keys = pygame.key.get_pressed()
                if keys[pygame.K_UP]:
                    paddle.move(1)
                if keys[pygame.K_DOWN]:
                    paddle.move(-1)
            elif PLAYER == "PSEUDO-AI":
                action = pseudo_ai(paddle, ball)
                paddle.move(action)
            elif PLAYER == "RL-AGENT":
                state = np.array([paddle.y, ball.x, ball.y, ball.dx, ball.dy])
                action = agent.get_action(state)

                # controller action
                controller_action = 1 if action == 1 else -1
                paddle.move(controller_action)

            ball.move()

            done = ball.x < 0

            collided = ball.check_collision(paddle)

            if PLAYER == "RL-AGENT":
                next_state = np.array([paddle.y, ball.x, ball.y, ball.dx, ball.dy])
                reward = 1 if collided else -1 if done else 0
                agent.update(state, action, reward, next_state, done)

            score += (1 if collided else 0)

            if done:
                generation += 1
                game_over()
                pygame.display.update()
                time.sleep(0.2)
                game_over_flag = True
                done = False

            paddle.draw()
            ball.draw()
            draw_header(score, generation)
            pygame.display.flip()

            clock.tick(FPS)

    pygame.quit()


if __name__ == "__main__":
    game_loop()

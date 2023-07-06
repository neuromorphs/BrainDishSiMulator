import pygame
import time
import cv2
import numpy as np

from controls import Paddle, Ball
import sys
sys.path.append("../../")
from models.rl_agents import DQNAgent

# Game parameters
WIDTH, HEIGHT = 600, 600
BALL_SIZE = 40
PADDLE_W, PADDLE_H = 40, 250
FONT_SIZE = 32

# Player type
PLAYER = "RL-AGENT"  # Choose between "AI" and "Human"

# Colors
WHITE = (255, 255, 255)
RED = (255, 0, 0)


def draw_env_status(screen):
    downscaled = cv2.resize(pygame.surfarray.array3d(screen), (40, 40))
    return downscaled


def draw_header(screen, font, score, generation):
    score_text = font.render(f"Generation {generation} - Score: {score}", True, WHITE)
    screen.blit(score_text, (WIDTH - score_text.get_width() - 10, 10))


def game_over(screen, font):
    game_over_text = font.render("Fail", True, RED)
    screen.blit(game_over_text, (WIDTH // 2 - game_over_text.get_width() // 2, HEIGHT // 2))


def pseudo_ai(paddle, ball):
    if paddle.y + PADDLE_H / 2 < ball.y:
        return -1
    else:
        return 1

def print_game_status(seed, fps, simulation_only, iteration, generation, score, reward=None, full=False):
    # print game status
    if full:
        print("Game status:")
        print("  - Player: {}".format(PLAYER))
        print("  - Seed: {}".format(seed))
        print("  - FPS: {}".format(fps))
        print("  - Simulation only: {}".format(simulation_only))
        print("  - Iteration: {}".format(iteration))
        print("  - Generation: {}".format(generation))
        print("  - Score: {}".format(score))
        print("  - Paddle size: {}x{}".format(PADDLE_W, PADDLE_H))
        print("  - Ball size: {}".format(BALL_SIZE))
        print("  - Screen size: {}x{}".format(WIDTH, HEIGHT))
        if reward is not None:
            print("  - Reward: {}".format(reward))
    else:
        if reward is None:
            print("Game status: It: {} - Gen: {} - Score: {}".format(iteration, generation, score))
        else:
            print("Game status: It: {} - Gen: {} - Score: {} - Reward: {}".format(iteration, generation, score, reward))


def game_loop(simulation_only=False, fps=60, verbose=False):
    # Pygame Initialization

    pygame.init()
    FONT = pygame.font.Font(None, FONT_SIZE)

    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()
    font = FONT

    score = 0
    running = True
    generation = 0

    # RL Agent
    agent = DQNAgent()
    seed = 0
    np.random.seed(seed)

    iteration = 0

    if verbose: print_game_status(seed, fps, simulation_only, iteration, generation, score, full=True)

    while running:
        done = False
        paddle = Paddle(PADDLE_W, PADDLE_H, screen)
        angle = np.random.choice([45, 135, 225, 315])
        dx = np.cos(angle)
        dy = np.sin(angle)
        ball = Ball(screen, dx, dy, BALL_SIZE)
        score = 0
        game_over_flag = False
        while not game_over_flag:
            if not simulation_only:
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
            next_state = np.array([paddle.y, ball.x, ball.y, ball.dx, ball.dy])
            reward = 1 if collided else -1 if done else 0

            if PLAYER == "RL-AGENT":
                agent.update(state, action, reward, next_state, done)

            score += (1 if collided else 0)

            if done:
                generation += 1
                if not simulation_only:
                    game_over(screen, font)
                    pygame.display.update()
                    time.sleep(0.2)
                game_over_flag = True
                done = False

            if verbose: print_game_status(seed, fps, simulation_only, iteration, generation, score, reward, full=False)

            if not simulation_only:
                paddle.draw()
                ball.draw()
                draw_header(screen, font, score, generation)
                pygame.display.flip()

                clock.tick(fps)

            iteration +=1

    pygame.quit()


if __name__ == "__main__":
    game_loop(simulation_only=False, fps=60, verbose=False)
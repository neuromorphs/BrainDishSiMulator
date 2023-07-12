import os

import pygame
import time
import cv2
import numpy as np

from controls import Paddle, Ball
import sys

sys.path.append("../../")
from models.rl_agents import DQNAgent, DQN8Agent, ConvDQNAgent, ConvDQNCaptureAgent, IFELSEAgent
from models.lif_agents import LIFDQNAgent, LIFELSEAgent, simple_LIF_else, simple_conductance_LIF
import time
import json
import argparse

# Game parameters
WIDTH, HEIGHT = 600, 600
BALL_SIZE = 40
BALL_SPEED = 1
PADDLE_W, PADDLE_H = 40, 250
FONT_SIZE = 32

# Player type
PLAYER = "PSEUDO-AI"  # Choose between DQN, DQN8, DQN8-onlypos, ConvDQN, Human and PSEUDO-AI

# Colors
WHITE = (255, 255, 255)
RED = (255, 0, 0)

FPS = 100

RESULT_FOLDER = None

CAPTURE_FOLDER = "captures"
os.makedirs(CAPTURE_FOLDER, exist_ok=True)


def draw_weights(screen, agent, w_size=50):
    if agent is None or not hasattr(agent, "get_weights"):
        return
    weights = agent.get_weights()
    # plot the weights of each layer on the right side of the screen
    for i in range(len(weights)):
        w = weights[i]
        # plot the weight matrix as an image
        wimg = w.detach().numpy()
        wimg = np.uint8(255 * (wimg - wimg.min()) / (wimg.max() - wimg.min()))
        wimg = cv2.resize(wimg, (w_size, w_size), interpolation=cv2.INTER_NEAREST)
        wimg = np.repeat(wimg[:, :, np.newaxis], 3, axis=2)
        wimg = pygame.surfarray.make_surface(wimg)
        screen.blit(wimg, (WIDTH - w_size, HEIGHT- (i+1)*w_size - (i+1)*10))





def get_40x40_env_status(screen):
    a = pygame.surfarray.array3d(screen)
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


def game_loop(seed, simulation_only=False, fps=60, save_capture=False, verbose=False):
    # Pygame Initialization
    pygame.init()
    FONT = pygame.font.Font(None, FONT_SIZE)

    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()
    font = FONT

    running = True
    num_episodes = 70

    # RL Agent
    num_inputs = 5
    num_actions = 2
    if PLAYER == "DQN":
        agent = DQNAgent(seed, num_inputs=num_inputs, num_outputs=num_actions)
    elif PLAYER == "ConvDQN":
        agent = ConvDQNAgent(seed, num_inputs=(40, 40, 1), num_outputs=num_actions)
    elif PLAYER == "ConvDQNCapture":
        agent = ConvDQNCaptureAgent(seed, num_inputs=(40, 40, 1), num_outputs=num_actions)
    elif PLAYER == "DQN8":
        agent = DQN8Agent(seed, num_inputs=num_inputs, num_outputs=num_actions)
    elif PLAYER == "DQN8-onlypos":
        agent = DQN8Agent(seed, num_inputs=3, num_outputs=num_actions)
    elif PLAYER == "DQN-LIF":
        agent = LIFDQNAgent(seed, num_inputs=5, num_outputs=num_actions, )
    elif PLAYER == "HUMAN":
        agent = None
    elif PLAYER == "PSEUDO-AI":
        agent = IFELSEAgent(seed)
    elif PLAYER == "LIFELSE":
        agent = LIFELSEAgent(seed, num_inputs=2, num_outputs=num_actions, hidden_units = [4], gamma=0.99,
                             tau_mem=5e-3, tau_syn=10e-3, lr=1e-2, simulation_timesteps=10, dt=1e-3)
    elif PLAYER == "SIMPLE_LIFELSE":
        agent = simple_LIF_else(current_scale=1)
    elif PLAYER == "SIMPLE_COBA" :
        agent = simple_conductance_LIF(conductance = 0.5*5)
    else:
        raise ValueError("Player type not supported")
    # save_config in json
    init_time = time.time()
    config = {
        "player": {"type": PLAYER,
                   "num_inputs": num_inputs,
                   "num_actions": num_actions,
                   "seed": seed},
        "fps": fps,
        "simulation_only": simulation_only,
        "num_episodes": num_episodes,
        "paddle_size": [PADDLE_W, PADDLE_H],
        "ball_size": BALL_SIZE,
        "ball_speed": BALL_SPEED,
        "screen_size": [WIDTH, HEIGHT],
        "start_time": init_time,
        "reward": "1 if collided else -1 if done else 0"

    }
    event_recording = []
    event_recording.append({"config": config})
    event_recording.append({"norm_timestamp": init_time - init_time, 'event': 'game begin'})  # first event

    iteration = 0
    episode = 0
    score = 0

    if verbose > 0: print_game_status(seed, fps, simulation_only, iteration, episode, score, full=True)

    # set seed for game reproducibility
    np.random.seed(0)

    while running:
        done = False
        paddle = Paddle(PADDLE_W, PADDLE_H, screen)
        angle = np.random.choice([45, 135, 225, 315])
        dx = 14
        dy = 1
        while dy<=1 and dy>=-1: # ensure bigger than 1
            tmp = np.random.uniform()
            dy = int(14.0 * tmp - 2.0)
        ball = Ball(screen, dx, dy, BALL_SIZE, ball_speed=BALL_SPEED)
        score = 0
        game_over_flag = False
        if verbose > 0: print("======", "Episode", episode, "======")

        if PLAYER == "ConvDQNCapture":
            last_capture = np.zeros((40, 40, 1))  # init capture system
            state = last_capture
        else:
            state = np.array([paddle.y, ball.x, ball.y, ball.dx, ball.dy])
            next_state = state

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
                
            elif PLAYER == "SIMPLE_LIFELSE" or PLAYER == "SIMPLE_COBA":
                action = agent.update(paddle.y, ball.y)
                paddle.move(action)

            elif PLAYER == "ConvDQN":
                previous_state = np.asarray(next_state)
                if len(previous_state.shape) != 1:
                    previous_state = next_state[1]
                state = [np.array([paddle.y, ball.x, ball.y, ball.dx, ball.dy]),
                         previous_state]
                action = agent.get_action(state)


            elif PLAYER == "ConvDQNCapture":
                state = capture - last_capture
                action = agent.get_action(state)
                last_capture = capture

                # controller action
                controller_action = 1 if action == 1 else -1
                paddle.move(controller_action)

            else:  # valid for DQN, DQN8, DQN8-onlypos
                state = np.array([paddle.y, ball.x, ball.y, ball.dx, ball.dy])
                action = agent.get_action(state)

                # controller action
                controller_action = 1 if action == 1 else -1
                paddle.move(controller_action)

            bounced = ball.move()
            done = ball.x < 0
            collided = ball.check_collision(paddle)

            if bounced:
                event_recording.append({"norm_timestamp": time.time() - init_time, 'event': 'ball bounce'})
            if done:
                event_recording.append({"norm_timestamp": time.time() - init_time, 'event': 'ball missed'})
                event_recording.append({"norm_timestamp": time.time() - init_time, 'event': 'motor layout: 0'})
            if collided:
                event_recording.append({"norm_timestamp": time.time() - init_time, 'event': 'ball return'})

            reward = 1 if collided else -1 if done else 0

            # update screen and scores
            score += (1 if collided else 0)
            if done or score > 100:  # episode is over if the score is greater than 100
                episode += 1
                if not simulation_only:
                    game_over(screen, font)
                    pygame.display.update()
                    time.sleep(0.2)
                game_over_flag = True

            if verbose > 1: print_game_status(seed, fps, simulation_only, iteration, episode, score, reward, full=False)
            if not simulation_only:
                paddle.draw()
                ball.draw()
                draw_header(screen, font, score, episode)
                draw_weights(screen, agent)
                pygame.display.flip()

                clock.tick(fps)

            if not simulation_only:
                capture = get_40x40_env_status(screen)
                # save capture as image
                if iteration % 100 == 0 and save_capture:
                    cv2.imwrite(os.path.join(CAPTURE_FOLDER, "capture_{}.png".format(iteration)), capture)

                # convert to grey scale
                capture = cv2.cvtColor(capture, cv2.COLOR_BGR2GRAY)
                # add channel dimension
                capture = np.expand_dims(capture, axis=-1)

            # RL Phase
            if PLAYER == "ConvDQNCapture":
                next_capture = get_40x40_env_status(screen)
                next_capture = cv2.cvtColor(next_capture, cv2.COLOR_BGR2GRAY)
                next_capture = np.expand_dims(next_capture, axis=-1)
                next_state = next_capture - capture
                agent.update(state, action, reward, next_state, done)


            elif PLAYER == "ConvDQN":
                next_state = [np.array([paddle.y, ball.x, ball.y, ball.dx, ball.dy]),
                              state[0]]
                agent.update(state, action, reward, next_state, done)
                
            elif PLAYER == "SIMPLE_LIFELSE" or PLAYER == "SIMPLE_COBA":
                pass 

            else:  # valid for DQN, DQN8, DQN8-onlypos
                print(action)
                next_state = np.array([paddle.y, ball.x, ball.y, ball.dx, ball.dy])
                agent.update(state, action, reward, next_state, done)

            iteration += 1

        if episode > num_episodes:
            break

    # save log in a file containing the player, the seed of initialization, the fps, the number of episodes, the number of iterations, the number of generations, the score, the reward, the paddle size, the ball size, the screen size
    log_filename = os.path.join(RESULT_FOLDER, "event_recording-{}.json".format(init_time))
    with open(log_filename, "w") as f:
        json.dump(event_recording, f)

    pygame.quit()


if __name__ == "__main__":
    # get arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--ball_speed", type=float, default=1.0, help="Ball speed")
    parser.add_argument("--player", type=str, default="DQN",
                        help="Player type: HUMAN, PSEUDO-AI, DQN, ConvDQN, ConvDQNCapture")
    parser.add_argument("--fps", type=int, default=100, help="Frames per second")
    parser.add_argument("--num_episodes", type=int, default=70, help="Number of episodes")
    parser.add_argument("--num_repeat", type=int, default=40, help="Number of repeats of the experiment")
    parser.add_argument("--simulation_only", type=bool, default=True, help="Simulation only")
    parser.add_argument("--save_capture", type=bool, default=False, help="Save capture")
    parser.add_argument("--verbose", type=int, default=1, help="Verbose level")
    args = parser.parse_args()

    BALL_SPEED = args.ball_speed
    PLAYER = args.player
    FPS = args.fps
    num_episodes = args.num_episodes
    simulation_only = False#args.simulation_only
    save_capture = args.save_capture
    verbose = args.verbose
    num_repeat = args.num_repeat

    # create result folder
    RESULT_FOLDER = "results_init_middle/{}/BALL_SPEED_{}".format(PLAYER, BALL_SPEED)
    os.makedirs(RESULT_FOLDER, exist_ok=True)

    for seed in range(num_repeat):
        game_loop(seed=seed, simulation_only=simulation_only, fps=FPS, save_capture=save_capture, verbose=1)

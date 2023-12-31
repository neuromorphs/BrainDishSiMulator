import os

import pygame
import time
import cv2
import numpy as np

from controls import Paddle, Ball
import sys

sys.path.append("../../")
from models.lif_agents import simple_LIF_else
from models.simon_LIF_agent import SimonAgent
from models.simon_LIF_hebbian import HebbianSimonAgent
from models.hugo_nonLIF_agent import HugoAgent
from models.torch_lif import LIF_FF
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
PLAYER = "LIFELSE"  # Choose between DQN, DQN8, DQN8-onlypos, ConvDQN, Human and PSEUDO-AI

# Colors
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)

FPS = 100
RESULT_FOLDER = None

R = 1
R_player = 2 # reward for winning against the player
P = -5
N = 0

CAPTURE_FOLDER = "captures"
os.makedirs(CAPTURE_FOLDER, exist_ok=True)


def game_loop(seed, simulation_only=False, fps=60, verbose=False, num_episodes = 70,
            player_paddle = False, record_video = False):
    
    # Pygame Initialization
    pygame.init()
    FONT = pygame.font.Font(None, FONT_SIZE)

    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()
    font = FONT

    running = True

    if PLAYER == "Simon": # SNN with gradient on membrane potentials (that's cheating)
        agent = SimonAgent(seed, num_inputs=2, hidden_units=[8,4], num_outputs=2, lr=1e-4,
                           reset=False, tau_mem=5e-3, tau_syn=4e-3, use_bias=True, dt=1e-3,
                    simulation_timesteps=10, gamma=0.99)
    elif PLAYER == "Hugo" : # Regular ol' dumb DQN network
        agent = HugoAgent(seed, num_inputs=2, num_outputs=2, hidden_units = [2], buffer_capacity=10000,
                            batch_size=1, gamma=0.99, lr=1e-3)
    elif PLAYER == "LIF_FF": # The future of AI, IF-ELSE statement with spikes
        agent = simple_LIF_else(current_scale=1)
    elif PLAYER == 'Hebbian_Simon' : # Dishbrain-like network, with hebbian learning on spikes (that's allowed)
        agent = HebbianSimonAgent(seed, num_inputs=2, hidden_units=[8,4], num_outputs=2, lr=1e-4,
                                reset=False, tau_mem=5e-3, tau_syn=4e-3, use_bias=True, dt=1e-3,
                                simulation_timesteps=10, gamma=0.99)
    else:
        raise ValueError("Player type not supported")
    
    # save_config in json
    init_time = time.time()
    config = {
        "player": {"type": PLAYER,
                "num_inputs": 2,
                "num_actions": 2,
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
    np.random.seed(seed)

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

        if player_paddle :
            right_paddle = Paddle(PADDLE_W, PADDLE_H, screen, side = 'right')
            
        state = np.array([paddle.y, ball.x, ball.y, ball.dx, ball.dy])
        next_state = state

        while not game_over_flag:
            if not simulation_only:
                screen.fill((0, 0, 0))

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            
            if player_paddle :
                keys = pygame.key.get_pressed()
                if keys[pygame.K_UP]:
                    right_paddle.move(1)
                if keys[pygame.K_DOWN]:
                    right_paddle.move(-1)
                    
            if PLAYER == "LIF_FF":
                action = agent.update(paddle.y, ball.y)
                paddle.move(action)
            else :
                state = np.array([paddle.y, ball.x, ball.y, ball.dx, ball.dy])
                action = agent.get_action(state)

            # controller action
            controller_action = 1 if action == 1 else -1
            paddle.move(controller_action)

            bounced = ball.move()
            done = ball.x < 0
            if player_paddle : player_lost = ball.x > (WIDTH-BALL_SIZE)
            
            collided = ball.check_collision(paddle)
            if player_paddle :
                collided_right = ball.check_collision(right_paddle)
                if collided_right:
                    event_recording.append({"norm_timestamp": time.time() - init_time, 'event': 'ball bounce'})

            if bounced:
                event_recording.append({"norm_timestamp": time.time() - init_time, 'event': 'ball bounce'})
            if done:
                event_recording.append({"norm_timestamp": time.time() - init_time, 'event': 'ball missed'})
                event_recording.append({"norm_timestamp": time.time() - init_time, 'event': 'motor layout: 0'})
            if collided:
                event_recording.append({"norm_timestamp": time.time() - init_time, 'event': 'ball return'})

            reward = N
            if collided:
                reward = R
            elif done:
                reward = P
            elif player_paddle :
                if player_lost:
                    reward = R_player
            else:
                reward = N

            # update screen and scores
            score += (1 if collided else 0)
            if done or score > 100:  # episode is over if the score is greater than 100
                episode += 1
                if not simulation_only:
                    game_over(screen, font)
                    pygame.display.update()
                    time.sleep(0.3)
                game_over_flag = True
            elif player_paddle :
                if player_lost:
                    episode +=1 
                    game_won(screen, font)
                    pygame.display.update()
                    time.sleep(0.3)
                    game_over_flag = True

            
            if not simulation_only:
                paddle.draw()
                if player_paddle : right_paddle.draw()
                ball.draw()
                draw_header(screen, font, score, episode)
                draw_spikes(screen, agent)
                draw_weights(screen, agent)
                pygame.display.flip()

                clock.tick(fps)

                if PLAYER == "LIF_FF":
                    pass
                else :
                    next_state = np.array([paddle.y, ball.x, ball.y, ball.dx, ball.dy])
                    agent.update(state, action, reward, next_state, done)

            if record_video :
                image = pygame.surfarray.array3d(pygame.display.get_surface())
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(f'./movie/frame_{iteration:06d}.png', image)
    
            iteration += 1
        if verbose > 1: print_game_status(seed, fps, simulation_only, iteration, episode, score, reward, full=False)

        if episode > num_episodes:
            break

    # save log in a file containing the player, the seed of initialization, the fps, the number of episodes, the number of iterations, the number of generations, the score, the reward, the paddle size, the ball size, the screen size
    log_filename = os.path.join(RESULT_FOLDER, "event_recording-{}.json".format(init_time))
    with open(log_filename, "w") as f:
        json.dump(event_recording, f)

    if hasattr(agent, "save_model"):
        agent.save_model('./models/{}.h5'.format(init_time))
    pygame.quit()


def draw_weights(screen, agent, w_size=50):
    if agent is None or not hasattr(agent, "get_weights"):
        return
    
    pygame.font.init()
    myfont = pygame.font.SysFont('Arial', 30)
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
        
        textsurface = myfont.render('W' + str(i), False, (255, 255, 255))
        # Adjust the y value here too.
        screen.blit(textsurface, (WIDTH - w_size*2, HEIGHT- (i+1)*w_size - (i+1)*10))  # Adjust position as needed
        
def draw_spikes(screen, agent, w_size = 50):
    if agent is None or not hasattr(agent, "get_spikes"):
        return
    spikes = agent.get_spikes()  # get binary spikes

    # Initialize a Pygame font
    pygame.font.init()
    myfont = pygame.font.SysFont('Arial', 30)

    for i in range(len(spikes)):
        # convert binary spike matrix to an image
        
        simg = np.uint8(255 * spikes[i])  # since it's binary, no need to normalize
        simg = cv2.resize(simg, (w_size, w_size), interpolation=cv2.INTER_NEAREST)
        simg = np.repeat(simg[:, :, np.newaxis], 3, axis=2)
        simg = pygame.surfarray.make_surface(simg)
        # Change the y value to be at the top of the screen.
        screen.blit(simg, (WIDTH - w_size, (i+1)*w_size + (i+1)*10))

        # Render the text "Layer i"
        textsurface = myfont.render('L' + str(i), False, (255, 255, 255))
        # Adjust the y value here too.
        screen.blit(textsurface, (WIDTH - w_size*2, (i+1)*w_size + 10))  # Adjust position as needed


def draw_header(screen, font, score, generation):
    score_text = font.render(f"Generation {generation} - Score: {score}", True, WHITE)
    screen.blit(score_text, (WIDTH - score_text.get_width() - 10, 10))


def game_over(screen, font):
    game_over_text = font.render("Model failed", True, GREEN)
    screen.blit(game_over_text, (WIDTH // 2 - game_over_text.get_width() // 2, HEIGHT // 2))
    
def game_won(screen, font):
    game_over_text = font.render("You failed", True, RED)
    screen.blit(game_over_text, (WIDTH // 2 - game_over_text.get_width() // 2, HEIGHT // 2))


def print_game_status(seed, fps, simulation_only, iteration, generation, score, reward=None, full=False,
                    player_paddle = False):
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
            
            
if __name__ == "__main__":
    # get arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--ball_speed", type=float, default=1.0, help="Ball speed")
    parser.add_argument("--player", type=str, default="DQN",
                        help="Player type: HUMAN, PSEUDO-AI, DQN, ConvDQN, ConvDQNCapture")
    parser.add_argument("--fps", type=int, default=100, help="Frames per second")
    parser.add_argument("--num_episodes", type=int, default=70, help="Number of episodes")
    parser.add_argument("--num_repeat", type=int, default=40, help="Number of repeats of the experiment")
    parser.add_argument("--verbose", type=int, default=1, help="Verbose level")
    parser.add_argument("--simulation_only", action='store_true', default=False, help="Simulation only")
    parser.add_argument("--record_video", action='store_true', default=False, help="Record video")
    parser.add_argument("--player_paddle", action='store_true', default=False, help="Player paddle")

    
    args = parser.parse_args()

    BALL_SPEED = args.ball_speed
    PLAYER = args.player
    FPS = args.fps
    num_episodes = args.num_episodes
    simulation_only = args.simulation_only
    verbose = args.verbose
    num_repeat = args.num_repeat
    record_video = args.record_video
    player_paddle = args.player_paddle
    
    # create result folder
    RESULT_FOLDER = "results_demo/{}/BALL_SPEED_{}".format(PLAYER, BALL_SPEED)
    os.makedirs(RESULT_FOLDER, exist_ok=True)
    for seed in range(num_repeat):
        game_loop(seed=seed, simulation_only=simulation_only, fps=FPS, verbose=verbose,
                num_episodes = num_episodes, player_paddle = player_paddle, record_video = record_video)

# modelo2.py
import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import PolicyNetwork
from helper import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001
GAMMA = 0.99  # Factor de descuento para refuerzo

class Agent:
    def __init__(self):
        self.n_games = 0
        self.gamma = GAMMA  # Descuento
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = PolicyNetwork(11, 256, 3)  # Ajusta el tamaño de la red
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LR)
    
    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),

            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),

            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),

            dir_l, dir_r, dir_u, dir_d,
            game.food.x < game.head.x,
            game.food.x > game.head.x,
            game.food.y < game.head.y,
            game.food.y > game.head.y
        ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward):
        self.memory.append((state, action, reward))

    def choose_action(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float)
        probs = self.model(state_tensor)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item(), action_dist.log_prob(action)


    def train_policy(self):
        if len(self.memory) < BATCH_SIZE:
            return

        batch = random.sample(self.memory, BATCH_SIZE)
        policy_losses = []
        rewards = []

        for state, action, reward in batch:
            state_tensor = torch.tensor(state, dtype=torch.float)
            reward_tensor = torch.tensor(reward, dtype=torch.float)
            probs = self.model(state_tensor)
            action_dist = torch.distributions.Categorical(probs)
            log_prob = action_dist.log_prob(torch.tensor(action))

            discounted_reward = reward_tensor * self.gamma
            policy_losses.append(-log_prob * discounted_reward)

        if policy_losses:
            policy_loss = torch.stack(policy_losses).sum()
        else:
            print("Warning: policy_losses is empty.")
            return


    def get_reward(self, game, done):
        return game.get_reward() if not done else -10

import matplotlib.pyplot as plt  # Ensure you import matplotlib

def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()

    while True:
        state_old = agent.get_state(game)
        action, log_prob = agent.choose_action(state_old)
        final_move = [0, 0, 0]
        final_move[action] = 1

        reward, done, score = game.play_step(final_move)
        agent.remember(state_old, action, reward)

        if done:
            game.reset()
            agent.n_games += 1
            agent.train_policy()

            if score > record:
                record = score
                # agent.model.save()

            print('Game', agent.n_games, 'Score', score, 'Record:', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            
            # Update plot without blocking
            plt.clf()  # Clear the previous plot
            plt.plot(plot_scores)
            plt.plot(plot_mean_scores)
            plt.pause(0.1)  # Pause to update the plot

if __name__ == '__main__':
    plt.ion()  # Turn on interactive plotting
    train()


import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from game import SnakeGameAI, Direction, Point
from helper import plot
import matplotlib.pyplot as plt

# Parámetros globales
MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001
GAMMA = 0.99  # Factor de descuento
TARGET_UPDATE = 10
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.05

# Modelo DuelingDQN
class DuelingDQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DuelingDQN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.V = nn.Linear(hidden_size, 1)  # Valor del estado V(s)
        self.A = nn.Linear(hidden_size, output_size)  # Ventaja de las acciones A(s, a)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        V = self.V(x)
        A = self.A(x)
        Q = V + (A - A.mean(dim=1, keepdim=True))  # Q(s, a) = V(s) + A(s, a) - mean(A)
        return Q

class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0.5  # Inicialmente explora mucho
        self.gamma = GAMMA
        self.memory = deque(maxlen=MAX_MEMORY)
        self.policy_net = DuelingDQN(11, 256, 3)  # Ajusta el tamaño de entrada y salida
        self.target_net = DuelingDQN(11, 256, 3)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=LR)
        self.loss_fn = nn.MSELoss()
        self.steps_done = 0

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
            # Danger straight
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),

            # Move direction
            dir_l, dir_r, dir_u, dir_d,
            
            # Food location
            game.food.x < head.x,  # food left
            game.food.x > head.x,  # food right
            game.food.y < head.y,  # food up
            game.food.y > head.y  # food down
        ]

        return np.array(state, dtype=int)

    def choose_action(self, state):
        self.epsilon = max(MIN_EPSILON, self.epsilon * EPSILON_DECAY)  # Decay dinámico
        if random.random() < self.epsilon:
            return random.randint(0, 2)  # Exploración
        else:
            state_tensor = torch.tensor(state, dtype=torch.float).unsqueeze(0)
            with torch.no_grad():
                action = self.policy_net(state_tensor).argmax().item()  # Explotación
            return action

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_policy(self):
        if len(self.memory) < BATCH_SIZE:
            return

        # Sample aleatorio del Replay Buffer
        batch = random.sample(self.memory, BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)

        import numpy as np
        states = torch.tensor(np.array(states), dtype=torch.float)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float)
        next_states = torch.tensor(next_states, dtype=torch.float)
        dones = torch.tensor(dones, dtype=torch.bool)

        # Q-values actuales
        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Acciones óptimas según la red de política
        next_actions = self.policy_net(next_states).argmax(1)

        # Evaluación de esas acciones con la red objetivo
        next_q_values = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)

        # Pérdida y optimización
        loss = self.loss_fn(q_values, target_q_values.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Actualización de la red objetivo
        if self.steps_done % TARGET_UPDATE == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        self.steps_done += 1

# Entrenamiento del agente
def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()

    while True:
        state_old = agent.get_state(game)
        action = agent.choose_action(state_old)
        final_move = [0, 0, 0]
        final_move[action] = 1

        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        agent.remember(state_old, action, reward, state_new, done)

        if done:
            game.reset()
            agent.n_games += 1
            agent.train_policy()

            if score > record:
                record = score
                torch.save(agent.policy_net.state_dict(), 'model_dueling_ddqn.pth')

            print(f'Game: {agent.n_games}, Score: {score}, Record: {record}')

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            
            plt.clf()
            plt.plot(plot_scores)
            plt.plot(plot_mean_scores)
            plt.pause(0.1)

if __name__ == '__main__':
    plt.ion()
    train()
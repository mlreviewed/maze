# 

import gym
from gym import spaces
import numpy as np
import random, copy


# 
class MazeEnv(gym.Env):
    def __init__(self, k=7, n_bots=2, local_grid_size=5, max_steps=1000):
        super(MazeEnv, self).__init__()
      
        self.k = k
        self.n_bots = n_bots
        self.local_grid_size = k
      
        self.max_steps = max_steps
        self.steps_taken = 0
        
        # Action space: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT
        self.action_space = spaces.Discrete(4)
        
        # Observation space for one agent:
        # - local_grid: (local_grid_size, local_grid_size, 4) (player, bots, points, walls)
        # - global_features: 3 (dist to nearest point, wall, other agent)
        # For bots, "other agent" is the player; for player, it's the nearest bot
        # For multi-bot, you can add more features (e.g., dist to other bots)
        self.observation_space = spaces.Dict({
            "local_grid": spaces.Box(low=0, high=1, shape=(local_grid_size, local_grid_size, 4), dtype=np.float32),
            "global_features": spaces.Box(low=0, high=k, shape=(3,), dtype=np.float32)
        })

        self.reset()
    
    def reset(self):
        
        self.grid = np.zeros((self.k, self.k))
        self.walls = np.zeros((self.k, self.k))
        self.points = np.zeros((self.k, self.k))
        self.player_pos = None
        self.bot_positions = []
        self.steps_taken = 0
        
        # Place walls
        # n_walls = random.randint(1, self.k)
        # for _ in range(n_walls):
        #     x, y = random.randint(0, self.k-1), random.randint(0, self.k-1)
        #     self.walls[x, y] = 2
        
        # # Place points - points are denoted with 1
        # n_points = random.randint(1, self.k)
        # for _ in range(n_points):
        #     while True:
        #         x, y = random.randint(0, self.k-1), random.randint(0, self.k-1)
        #         if self.walls[x, y] == 0:
        #             self.points[x, y] = 1
        #             break
        
        # # Place player - player is denoted with 0
        # while True:
        #     x, y = random.randint(0, self.k-1), random.randint(0, self.k-1)
        #     if self.walls[x, y] == 0 and self.points[x, y] == 0:
        #         self.player_pos = (x, y)
        #         break
      
        # # Place bots - bots are denoted with 3
        # for _ in range(self.n_bots):
        #     while True:
        #         x, y = random.randint(0, self.k-1), random.randint(0, self.k-1)
        #         if (x, y) != self.player_pos and self.walls[x, y] == 0 and self.points[x, y] == 0:
        #             self.bot_positions.append((x, y))
        #             break
        
        # return self._get_obs_for_agents()
    
    def _get_obs(self):
        k = self.k
        local_grid_size = self.local_grid_size
        px, py = self.player_pos
    
        # --- 1. Build the 4 channels ---
        player_channel = np.zeros((k, k), dtype=np.float32)
        player_channel[px, py] = 1.0
        
        bot_channel = np.zeros((k, k), dtype=np.float32)
        for bx, by in self.bot_positions:
            bot_channel[bx, by] = 1.0
        
        points_channel = self.points.astype(np.float32)
        walls_channel = self.walls.astype(np.float32)
    
        # Stack into (k, k, 4)
        full_grid = np.stack([player_channel, bot_channel, points_channel, walls_channel], axis=-1)
    
        # --- 2. Extract local grid centered on player ---
        pad = local_grid_size // 2
        padded_grid = np.pad(full_grid, ((pad, pad), (pad, pad), (0,0)), mode='constant')
        local_grid = padded_grid[px:px+local_grid_size, py:py+local_grid_size, :]
    
        # --- 3. Compute global features ---
        # Distance to nearest point
        point_indices = np.argwhere(self.points > 0)
        if len(point_indices) > 0:
            dist_to_point = np.min(np.sum(np.abs(point_indices - np.array([px, py])), axis=1))
        else:
            dist_to_point = float(k)
    
        # Distance to nearest wall
        wall_indices = np.argwhere(self.walls > 0)
        if len(wall_indices) > 0:
            dist_to_wall = np.min(np.sum(np.abs(wall_indices - np.array([px, py])), axis=1))
        else:
            dist_to_wall = float(k)
    
        # Distance to nearest bot
        if self.bot_positions:
            dists_to_bots = [abs(bx - px) + abs(by - py) for bx, by in self.bot_positions]
            dist_to_bot = min(dists_to_bots)
        else:
            dist_to_bot = float(k)
    
        global_features = np.array([dist_to_point, dist_to_wall, dist_to_bot], dtype=np.float32)
    
        # --- 4. Return as dict ---
        return {
            "local_grid": local_grid,
            "global_features": global_features
        }
    
    # def _get_obs_for_bot(self, bot_idx):
    #   pass
  
    # def _get_obs_for_agent(self, agent_idx, agent_pos):
    #     # agent_idx: 0=player, 1..n_bots=bots
    #     # agent_pos: (x, y) position of the agent
        
    #     # --- Local grid ---
    #     half = self.local_grid_size // 2
    #     local_grid = np.zeros((self.local_grid_size, self.local_grid_size, 4))

    #     for i in range(self.local_grid_size):
    #         for j in range(self.local_grid_size):
    #             x = agent_pos[0] + i - half
    #             y = agent_pos[1] + j - half
    #             if 0 <= x < self.k and 0 <= y < self.k:
    #                 # Player channel
    #                 if agent_idx == 0:  # player sees itself as "player"
    #                     local_grid[i, j, 0] = (x == agent_pos[0] and y == agent_pos[1])
    #                 else:  # bots see the player as "player"
    #                     local_grid[i, j, 0] = (x == self.player_pos[0] and y == self.player_pos[1])
    #                 # Bots channel
    #                 if agent_idx == 0:  # player sees all bots
    #                     for bot_pos in self.bot_positions:
    #                         if x == bot_pos[0] and y == bot_pos[1]:
    #                             local_grid[i, j, 1] = 1
    #                 else:  # bots see other bots (except themselves)
    #                     for idx, bot_pos in enumerate(self.bot_positions):
    #                         if idx != agent_idx-1 and x == bot_pos[0] and y == bot_pos[1]:
    #                             local_grid[i, j, 1] = 1
    #                 # Points channel
    #                 local_grid[i, j, 2] = self.points[x, y]
    #                 # Walls channel
    #                 local_grid[i, j, 3] = self.walls[x, y]

    #     # --- Global features ---
    #     # 1. Distance to nearest point
    #     min_point_dist = self.k * 2  # large value
    #     for x in range(self.k):
    #         for y in range(self.k):
    #             if self.points[x, y] == 1:
    #                 dist = abs(x - agent_pos[0]) + abs(y - agent_pos[1])
    #                 if dist < min_point_dist:
    #                     min_point_dist = dist

    #     # 2. Distance to nearest wall
    #     min_wall_dist = self.k * 2
    #     for x in range(self.k):
    #         for y in range(self.k):
    #             if self.walls[x, y] == 1:
    #                 dist = abs(x - agent_pos[0]) + abs(y - agent_pos[1])
    #                 if dist < min_wall_dist:
    #                     min_wall_dist = dist

    #     # 3. Distance to nearest other agent
    #     min_agent_dist = self.k * 2
    #     if agent_idx == 0:  # player: distance to nearest bot
    #         for bot_pos in self.bot_positions:
    #             dist = abs(bot_pos[0] - agent_pos[0]) + abs(bot_pos[1] - agent_pos[1])
    #             if dist < min_agent_dist:
    #                 min_agent_dist = dist
    #     else:  # bot: distance to player
    #         dist = abs(self.player_pos[0] - agent_pos[0]) + abs(self.player_pos[1] - agent_pos[1])
    #         min_agent_dist = dist

    #     global_features = np.array([min_point_dist, min_wall_dist, min_agent_dist], dtype=np.float32)

    #     return {
    #         "local_grid": local_grid,
    #         "global_features": global_features
    #     }
    
    # def _get_obs_for_agents(self):
    #     # Get observation for player (agent_idx=0)
    #     player_obs = self._get_obs_for_agent(0, self.player_pos)
    #     # Get observation for each bot (agent_idx=1..n_bots)
    #     bot_obs = [self._get_obs_for_agent(i+1, self.bot_positions[i]) for i in range(self.n_bots)]
    #     return [player_obs] + bot_obs
    
    def _is_valid_move(self, pos):
        x, y = pos
        return (0 <= x < self.k and 0 <= y < self.k and self.walls[x, y] == 0)
    
    def _move(self, pos, action):
        x, y = pos
        if action == 0:   # UP
            x = max(0, x-1)
        elif action == 1: # DOWN
            x = min(self.k-1, x+1)
        elif action == 2: # LEFT
            y = max(0, y-1)
        elif action == 3: # RIGHT
            y = min(self.k-1, y+1)
        return (x, y)
    
    def step(self, actions):
        if len(actions) != self.n_bots + 1:
            raise ValueError(f"Expected {self.n_bots + 1} actions, got {len(actions)}")
    
        # 1. Compute intended new positions for all agents
        # Player
        intended_player_pos = self._move(self.player_pos, actions[0])
        if not self._is_valid_move(intended_player_pos):
            intended_player_pos = self.player_pos
    
        # Bots
        intended_bot_positions = []
        for i in range(self.n_bots):
            intended_pos = self._move(self.bot_positions[i], actions[i+1])
            if not self._is_valid_move(intended_pos):
                intended_pos = self.bot_positions[i]
            intended_bot_positions.append(intended_pos)
    
        # 2. Update positions (bots can overlap, player and bots can overlap)
        self.player_pos = intended_player_pos
        self.bot_positions = intended_bot_positions
    
        # 3. Point collection (player only)
        x, y = self.player_pos
        point_reward = 0
        if self.points[x, y] == 1:
            self.points[x, y] = 0
            point_reward = 1
        
        # 4. Check if player is caught by any bot
        caught = any(bot_pos == self.player_pos for bot_pos in self.bot_positions)
        done = caught or np.sum(self.points) == 0 or self.steps_taken >= self.max_steps
    
        # 5. Reward (player only)
        reward = point_reward
        if caught:
            reward = -10
    
        self.steps_taken += 1
    
        # 6. Build observations for all agents (example for player only)
        obs = self._get_obs_for_player()
        info = {
            "player_pos": self.player_pos,
            "bot_positions": self.bot_positions.copy(),
            "points_remaining": int(np.sum(self.points)),
            "caught": caught,
        }
    
        return obs, reward, done, info




class MazeGenerator:
    def __init__(self, k, population_size=20, wall_density=0.3):
        self.k = k
        self.population_size = population_size
        self.wall_density = wall_density
    
    def random_maze(self):
        # Randomly place walls with a given density, ensuring start and end are open
        maze = np.zeros((self.k, self.k), dtype=np.int32)
        for i in range(self.k):
            for j in range(self.k):
                if random.random() < self.wall_density:
                    maze[i, j] = 1
        # Ensure start and end are open
        maze[0, 0] = 0
        maze[self.k-1, self.k-1] = 0
        return maze

    def initialize_population(self):
        return [self.random_maze() for _ in range(self.population_size)]

    def mutate(self, maze, mutation_rate=0.01):
        new_maze = maze.copy()
        for i in range(self.k):
            for j in range(self.k):
                if (i, j) not in [(0,0), (self.k-1, self.k-1)] and random.random() < mutation_rate:
                    new_maze[i, j] = 1 - new_maze[i, j]
        return new_maze

    def crossover(self, maze1, maze2):
        # Simple row-wise crossover
        crossover_point = random.randint(1, self.k-2)
        child = np.vstack((maze1[:crossover_point], maze2[crossover_point:]))
        return child

    def fitness(self, maze, metric_fn):
        # metric_fn should be a function that evaluates the maze and returns a score
        return metric_fn(maze)

    def evolve(self, generations, metric_fn, mutation_rate=0.01, elite_fraction=0.2):
        population = self.initialize_population()
        for gen in range(generations):
            fitness_scores = [self.fitness(m, metric_fn) for m in population]
            # Select elites
            elite_count = max(1, int(self.population_size * elite_fraction))
            elites = [population[i] for i in np.argsort(fitness_scores)[-elite_count:]]
            # Breed new population
            new_population = elites.copy()
            while len(new_population) < self.population_size:
                parents = random.sample(elites, 2)
                child = self.crossover(parents[0], parents[1])
                child = self.mutate(child, mutation_rate)
                new_population.append(child)
            population = new_population
        # Return the best maze
        best_idx = np.argmax([self.fitness(m, metric_fn) for m in population])
        return population[best_idx]

import scipy.ndimage

def path_length_metric(maze):
    # Use BFS to find shortest path from (0,0) to (k-1,k-1)
    k = maze.shape[0]
    visited = np.zeros_like(maze)
    queue = [(0, 0, 0)]  # (x, y, distance)
    while queue:
        x, y, dist = queue.pop(0)
        if (x, y) == (k-1, k-1):
            return dist
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            nx, ny = x+dx, y+dy
            if 0 <= nx < k and 0 <= ny < k and maze[nx, ny] == 0 and not visited[nx, ny]:
                visited[nx, ny] = 1
                queue.append((nx, ny, dist+1))
    return 0  # Unsolvable

# 1. Evolve a maze
generator = MazeGenerator(k=7, population_size=30)
evolved_maze = generator.evolve(
    generations=50,
    metric_fn=path_length_metric,
    mutation_rate=0.05,
    elite_fraction=0.2
)

# 2. Instantiate the environment with the evolved maze
env = MazeEnv(k=7, n_bots=2, local_grid_size=5, max_steps=1000)
env.walls = evolved_maze.copy()  # Set the maze walls

# 3. Optionally, reset the environment to start a new episode
obs = env.reset()
env.walls = evolved_maze.copy()  # Re-apply the evolved maze after reset


print("Result:")
for row in env.walls:
    print("".join(['#' if cell else '.' for cell in row.astype(int)]))

# Now you can use env as usual


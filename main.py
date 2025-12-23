import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
GRID_SIZE = 6
NUM_AGENTS = 2
NUM_ACTIONS = 5  # Up, Down, Left, Right, Stay
HIDDEN_SIZE = 64
LR_ACTOR = 0.0003
LR_CRITIC = 0.001
GAMMA = 0.99        # Discount factor
GAE_LAMBDA = 0.95   # GAE smoothing
CLIP_EPSILON = 0.2  # PPO Clipping range
UPDATE_EPOCHS = 4
BATCH_SIZE = 32
MAX_STEPS = 50      # Max steps per episode
TOTAL_EPISODES = 600 

# CUDA SETUP
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- THE ENVIRONMENT ---
class GridWorldEnv:
    def __init__(self):
        self.grid_size = GRID_SIZE
        self.target_pos = None
        self.agent_positions = [None] * NUM_AGENTS
        
    def reset(self):
        # Randomize target
        self.target_pos = np.random.randint(0, self.grid_size, size=2)
        
        # Randomize agents (ensure they don't start on target)
        for i in range(NUM_AGENTS):
            while True:
                pos = np.random.randint(0, self.grid_size, size=2)
                if not np.array_equal(pos, self.target_pos):
                    self.agent_positions[i] = pos
                    break
        return self._get_obs(), self._get_global_state()

    def _get_obs(self):
        # Decentralized Observation: [Agent_X, Agent_Y, Target_X, Target_Y]
        observations = []
        for i in range(NUM_AGENTS):
            # Normalize coordinates to 0-1 range for better Neural Net performance
            obs = np.concatenate([
                self.agent_positions[i] / self.grid_size, 
                self.target_pos / self.grid_size
            ])
            observations.append(obs)
        return np.array(observations, dtype=np.float32)

    def _get_global_state(self):
        # Centralized State: [Ag1_X, Ag1_Y, Ag2_X, Ag2_Y, Target_X, Target_Y]
        # The Critic sees EVERYTHING
        flat_agents = np.concatenate(self.agent_positions) / self.grid_size
        normalized_target = self.target_pos / self.grid_size
        state = np.concatenate([flat_agents, normalized_target])
        return np.array(state, dtype=np.float32)

    def step(self, actions):
        rewards = np.zeros(NUM_AGENTS)
        done = False
        
        # Movement mapping
        # 0: Up, 1: Down, 2: Left, 3: Right, 4: Stay
        moves = {
            0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1), 4: (0, 0)
        }

        for i, action in enumerate(actions):
            move = moves[action]
            new_pos = self.agent_positions[i] + move
            
            # Boundary Check
            new_pos = np.clip(new_pos, 0, self.grid_size - 1)
            self.agent_positions[i] = new_pos
            
            # Distance Calculation (Manhattan)
            dist = np.sum(np.abs(self.agent_positions[i] - self.target_pos))
            
            # Reward Structure
            if np.array_equal(self.agent_positions[i], self.target_pos):
                rewards[i] += 10.0 # Big reward for reaching goal
            else:
                rewards[i] -= 0.1 * dist # Shaping: Penalty based on distance (encourages moving closer)
                rewards[i] -= 0.01 # Time penalty
                
        done = any(np.array_equal(pos, self.target_pos) for pos in self.agent_positions)
        
        return self._get_obs(), self._get_global_state(), rewards, done

# --- NETWORKS (ACTOR & CRITIC) ---
class Actor(nn.Module):
    """Decentralized: Takes local observation, outputs action probabilities."""
    def __init__(self, obs_dim, action_dim):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, HIDDEN_SIZE),
            nn.Tanh(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.Tanh(),
            nn.Linear(HIDDEN_SIZE, action_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.net(x)

class Critic(nn.Module):
    """Centralized: Takes GLOBAL state, outputs Value (V)."""
    def __init__(self, global_state_dim):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(global_state_dim, HIDDEN_SIZE),
            nn.Tanh(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.Tanh(),
            nn.Linear(HIDDEN_SIZE, 1)
        )

    def forward(self, x):
        return self.net(x)

# --- MAPPO AGENT ---
class MAPPOAgent:
    def __init__(self):
        # Obs: [x, y, tx, ty] = 4
        self.actor = Actor(obs_dim=4, action_dim=NUM_ACTIONS).to(device)
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=LR_ACTOR)
        
        # Global State: [x1, y1, x2, y2, tx, ty] = 6
        self.critic = Critic(global_state_dim=6).to(device)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=LR_CRITIC)
        
    def get_action(self, obs):
        obs_tensor = torch.FloatTensor(obs).to(device)
        probs = self.actor(obs_tensor)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

    def evaluate_actions(self, obs, actions):
        # Used during update loop
        probs = self.actor(obs)
        dist = torch.distributions.Categorical(probs)
        action_log_probs = dist.log_prob(actions)
        dist_entropy = dist.entropy()
        return action_log_probs, dist_entropy

# --- MAPPO UPDATE LOGIC ---
def compute_gae(next_value, rewards, masks, values):
    """Generalized Advantage Estimation"""
    values = values + [next_value]
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + GAMMA * values[step + 1] * masks[step] - values[step]
        gae = delta + GAMMA * GAE_LAMBDA * masks[step] * gae
        returns.insert(0, gae + values[step])
    return returns

def ppo_update(agent_system, memory):
    # Unpack Memory
    obs_batch = torch.FloatTensor(np.array(memory['obs'])).to(device)
    state_batch = torch.FloatTensor(np.array(memory['state'])).to(device)
    actions_batch = torch.LongTensor(np.array(memory['actions'])).to(device)
    logprobs_batch = torch.FloatTensor(np.array(memory['logprobs'])).to(device)
    returns_batch = torch.FloatTensor(np.array(memory['returns'])).to(device)
    
    # Optimize for N epochs
    for _ in range(UPDATE_EPOCHS):
        # Get current V(s) estimates and Policy distributions
        # 'agent_system' as the shared brain for both agents.
        
        current_values = agent_system.critic(state_batch).squeeze()
        action_log_probs, dist_entropy = agent_system.evaluate_actions(obs_batch, actions_batch)
        
        # Calculate Ratios and Advantages
        advantages = returns_batch - current_values.detach()
        # Normalize advantages (trick for training stability)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)
        
        ratios = torch.exp(action_log_probs - logprobs_batch)
        
        # Surrogate Loss (The PPO Clipping)
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1.0 - CLIP_EPSILON, 1.0 + CLIP_EPSILON) * advantages
        
        loss_actor = -torch.min(surr1, surr2).mean() - 0.01 * dist_entropy.mean()
        loss_critic = 0.5 * nn.MSELoss()(current_values, returns_batch)
        
        # Backpropagation
        agent_system.optimizer_actor.zero_grad()
        loss_actor.backward()
        agent_system.optimizer_actor.step()
        
        agent_system.optimizer_critic.zero_grad()
        loss_critic.backward()
        agent_system.optimizer_critic.step()

# --- VISUALIZATION ---
def visualize_episodes(env, agent_system, num_episodes=5, realtime=True, pause_between_eps=0.8):
    agent_system.actor.eval()
    agent_system.critic.eval()

    fig, ax = plt.subplots(figsize=(5, 5))

    def to_xy(pos_rc):
        return int(pos_rc[1]), int(pos_rc[0])

    def draw(ep, step):
        ax.clear()
        ax.set_xlim(-0.5, GRID_SIZE - 0.5)
        ax.set_ylim(-0.5, GRID_SIZE - 0.5)
        ax.set_xticks(range(GRID_SIZE))
        ax.set_yticks(range(GRID_SIZE))
        ax.grid(True)

        ax.invert_yaxis()

        # Target
        tx, ty = to_xy(env.target_pos)
        ax.scatter(tx, ty, c="red", s=200, marker="*", label="Target")

        # Agents
        for i in range(NUM_AGENTS):
            x, y = to_xy(env.agent_positions[i])
            color = "blue" if i == 0 else "green"
            ax.scatter(x, y, c=color, s=100, label=f"Agent {i+1}")

        ax.legend(loc="upper right")

        winners = [i+1 for i, p in enumerate(env.agent_positions) if np.array_equal(p, env.target_pos)]
        if winners:
            ax.set_title(f"Episode {ep}/{num_episodes} | ✅ Reached by {winners} | Step {step}")
        else:
            ax.set_title(f"Episode {ep}/{num_episodes} | Step {step}/{MAX_STEPS}")

        fig.canvas.draw()

    for ep in range(1, num_episodes + 1):
        obs, state = env.reset()
        draw(ep, step=0)
        if realtime:
            plt.pause(0.25)

        for step in range(1, MAX_STEPS + 1):
            # choose actions
            actions = []
            with torch.no_grad():
                for i in range(NUM_AGENTS):
                    act, _ = agent_system.get_action(obs[i])
                    actions.append(act)

            # STEP FIRST
            obs, state, _, done = env.step(actions)

            draw(ep, step)
            if realtime:
                plt.pause(0.25)

            if done:
                if realtime:
                    plt.pause(pause_between_eps)
                break

    if realtime:
        plt.show()
    else:
        plt.close(fig)

# --- MAIN TRAINING LOOP ---
def train():
    env = GridWorldEnv()
    
    # Parameter Sharing: One network controls both agents
    mappo_brain = MAPPOAgent()
    
    print("Starting Training...")
    history_rewards = []

    for episode in range(1, TOTAL_EPISODES + 1):
        obs, state = env.reset()
        
        # Buffers for this episode
        memory = {'obs': [], 'state': [], 'actions': [], 'logprobs': [], 'rewards': [], 'masks': [], 'values': []}
        ep_reward = 0
        
        for step in range(MAX_STEPS):
            actions_list = []
            logprobs_list = []
            
            # Action Selection (Decentralized)
            for i in range(NUM_AGENTS):
                # Actor sees individual obs
                action, log_prob = mappo_brain.get_action(obs[i])
                actions_list.append(action)
                logprobs_list.append(log_prob)
            
            # Critic sees global state (only need value once per step for the shared brain)
            state_tensor = torch.FloatTensor(state).to(device)
            value = mappo_brain.critic(state_tensor).item()
            
            # Step Environment
            next_obs, next_state, rewards, _ = env.step(actions_list)
            
            # Store Data
            # (Agent 1's experience) and (Agent 2's experience) as two data points in the batch.
            for i in range(NUM_AGENTS):
                memory['obs'].append(obs[i])
                memory['state'].append(state) # Both agents share the global state context
                memory['actions'].append(actions_list[i])
                memory['logprobs'].append(logprobs_list[i].item())
                memory['rewards'].append(rewards[i])
                memory['masks'].append(1)
                memory['values'].append(value)
            
            obs = next_obs
            state = next_state
            ep_reward += sum(rewards)
            
            # Stop if max steps (implied by loop) or solved
            if all([np.array_equal(pos, env.target_pos) for pos in env.agent_positions]):
                break

        # Compute Returns (GAE)
        next_value = mappo_brain.critic(torch.FloatTensor(next_state).to(device)).item()
        returns = compute_gae(next_value, memory['rewards'], memory['masks'], memory['values'])
        memory['returns'] = returns
        
        # Update Networks
        ppo_update(mappo_brain, memory)
        
        history_rewards.append(ep_reward)
        if episode % 50 == 0:
            avg_rew = np.mean(history_rewards[-50:])
            print(f"Episode {episode}: Average Reward: {avg_rew:.2f}")

    print("Training Complete.")
    return mappo_brain

# --- RUNNER ---
if __name__ == "__main__":
    # Train
    trained_brain = train()
    
    # Evaluate & Visualize
    print("\nStarting Real-time Visualization...")
    # Create a fresh env for viz
    viz_env = GridWorldEnv()
    visualize_episodes(viz_env, trained_brain, num_episodes=5, realtime=True)

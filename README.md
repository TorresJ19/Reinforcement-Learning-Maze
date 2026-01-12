# Maze Maker & RL Algorithm Tester

**Enterprise-Grade Reinforcement Learning Platform**

A sophisticated maze creation and testing platform that enables users to design custom mazes and evaluate reinforcement learning algorithms (PPO and DQN) through comprehensive performance analysis and real-time visualization.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture & Design Logic](#architecture--design-logic)
3. [Core Components](#core-components)
4. [Reinforcement Learning Algorithms](#reinforcement-learning-algorithms)
5. [Environment Design](#environment-design)
6. [User Interface](#user-interface)
7. [Quickstart](#quickstart)
8. [Usage Guide](#usage-guide)
9. [Technical Details](#technical-details)

---

## Overview

This platform provides a complete workflow for:
- **Maze Design**: Interactive editor for creating custom maze layouts
- **Algorithm Testing**: Compare PPO and DQN performance on identical mazes
- **Real-time Visualization**: Watch agents learn and adapt in real-time
- **Performance Analytics**: Comprehensive metrics and statistics

---

## Architecture & Design Logic

### Why This Architecture?

The system is designed with **separation of concerns** and **modularity** in mind:

1. **Environment Layer** (`src/envs/`): Implements the Gymnasium interface, making it compatible with standard RL libraries. The environment is **stateless** and **deterministic** when seeded, ensuring reproducibility.

2. **Agent Layer** (`src/agents/`): Each algorithm is self-contained with its own configuration, network, and update logic. This allows easy addition of new algorithms without modifying existing code.

3. **Model Layer** (`src/models/`): Shared neural network architectures reduce code duplication. The ActorCritic network serves both policy and value estimation, while DuelingMLP provides Q-value estimation with advantage decomposition.

4. **Utility Layer** (`src/utils/`): Specialized buffers (RolloutBuffer for PPO, ReplayBuffer for DQN) match each algorithm's data requirements.

5. **Application Layer** (`scripts/`): User-facing tools that orchestrate the lower layers, providing intuitive interfaces for maze creation and algorithm testing.

### Design Principles

- **Reproducibility**: All randomness is seeded (Python, NumPy, PyTorch, environment)
- **Extensibility**: New algorithms can be added by implementing a standard interface
- **Performance**: Efficient tensor operations, hardware acceleration support
- **Usability**: Intuitive UI with real-time feedback

---

## Core Components

### 1. Maze Environment (`src/envs/maze_env.py`)

**Purpose**: Provides a standardized RL environment following the Gymnasium API.

**Key Design Decisions**:

- **Observation Space**: 4-channel grid representation
  - Channel 0: Agent position (1.0 where agent is, 0.0 elsewhere)
  - Channel 1: Walls (1.0 for walls, 0.0 for empty)
  - Channel 2: Obstacles (1.0 for obstacles, 0.0 for empty)
  - Channel 3: Goal position (1.0 where goal is, 0.0 elsewhere)
  
  **Why 4 channels?** This provides the agent with complete spatial information in a format that CNNs can efficiently process. The agent can see its position, obstacles, and goal simultaneously.

- **Action Space**: Discrete 4 actions (Up, Down, Left, Right)
  - Simple and intuitive for grid-based navigation
  - Matches the problem domain perfectly

- **Reward Structure**:
  ```
  Base reward per step: -0.1 (encourages efficiency)
  Goal reached: +10.0 (primary objective)
  Collision penalty: -2.0 (discourages hitting walls/obstacles)
  Efficiency bonus: +5.0 * (1.0 + steps_saved * 0.1) (encourages path optimization)
  Exploration bonus: +0.1 for visiting new positions
  Position repeat penalty: -0.5 * (repeats - 5) (prevents getting stuck)
  ```
  
  **Why this reward structure?**
  - **Step penalty (-0.1)**: Encourages finding shorter paths without being too punitive
  - **Goal reward (+10.0)**: Large positive signal for success
  - **Collision penalty (-2.0)**: Moderate penalty that doesn't terminate episodes, allowing learning from mistakes
  - **Efficiency bonus**: Rewards finding shorter paths than previous best, encouraging continuous optimization
  - **Exploration bonus**: Small reward for visiting new cells, encouraging exploration
  - **Repeat penalty**: Prevents the agent from getting stuck in loops

- **Non-terminal Collisions**: Collisions don't end episodes. This allows the agent to learn from mistakes rather than restarting, which is crucial for learning in complex mazes.

- **Reward Shaping**: Optional Manhattan distance-based shaping provides intermediate rewards for moving closer to the goal, helping with sparse reward problems.

### 2. PPO Agent (`src/agents/ppo.py`)

**Purpose**: Implements Proximal Policy Optimization, an on-policy actor-critic algorithm.

**Algorithm Logic**:

1. **Actor-Critic Architecture**: 
   - **Actor**: Outputs action probabilities (policy π)
   - **Critic**: Estimates state value V(s)
   - **Why shared backbone?** The early layers learn common features (maze structure, spatial relationships), reducing computation and improving sample efficiency.

2. **Rollout Collection**:
   - Agent collects experiences for 64 steps (rollout horizon)
   - Stores: observations, actions, log probabilities, rewards, dones, values
   - **Why 64 steps?** Balance between sample efficiency (longer rollouts) and update frequency (shorter rollouts). 64 provides frequent updates while maintaining stable gradients.

3. **Advantage Estimation (GAE - Generalized Advantage Estimation)**:
   ```
   δ_t = r_t + γ * V(s_{t+1}) - V(s_t)  (TD error)
   A_t = δ_t + (γλ) * δ_{t+1} + (γλ)² * δ_{t+2} + ...
   ```
   - **Why GAE?** Reduces variance in advantage estimates while maintaining low bias
   - **λ = 0.95**: Balances bias-variance tradeoff

4. **Policy Update (Clipped Surrogate Objective)**:
   ```
   L^CLIP(θ) = E[min(r(θ) * A, clip(r(θ), 1-ε, 1+ε) * A)]
   ```
   - **Why clipping?** Prevents large policy updates that could destabilize training
   - **ε = 0.2**: Standard PPO clipping range
   - **Why ratio r(θ)?** Measures how much the new policy differs from the old policy

5. **Value Function Update**:
   ```
   L^VF(θ) = (V_θ(s) - V_target)²
   ```
   - Fits value function to actual returns
   - Helps with advantage estimation accuracy

6. **Entropy Bonus**:
   ```
   L = L^CLIP - c_v * L^VF + c_e * H(π)
   ```
   - **Why entropy?** Encourages exploration by penalizing overly confident policies
   - **c_e = 0.3 (initial)**: Starts high for exploration, decays to 0.05 for exploitation
   - **Adaptive entropy**: Increases when agent gets stuck, decreases as training progresses

7. **Mini-batch Updates**:
   - Rollout buffer (64 steps) split into mini-batches (16 steps each)
   - **Why mini-batches?** More stable gradients, better generalization, allows multiple passes
   - **4 mini-batches per epoch**: Provides 4 gradient updates per rollout

**Hyperparameters**:
- Learning rate: 3e-4 (standard for Adam optimizer)
- Gamma (discount): 0.99 (long-term planning)
- GAE lambda: 0.95 (bias-variance tradeoff)
- Clip coefficient: 0.2 (prevents large updates)
- Update epochs: 4 (multiple passes over data)
- Mini-batch size: 16 (optimized for rollout size 64)

### 3. DQN Agent (`src/agents/dqn.py`)

**Purpose**: Implements Deep Q-Network, an off-policy value-based algorithm.

**Algorithm Logic**:

1. **Q-Learning with Function Approximation**:
   - Learns Q(s, a) = expected future reward from state s taking action a
   - Uses neural network to approximate Q-function
   - **Why Q-learning?** Can learn from off-policy data (past experiences)

2. **Experience Replay**:
   - Stores experiences in a buffer (capacity: 50,000)
   - Samples random batches for training
   - **Why experience replay?** 
     - Breaks correlation between consecutive samples
     - Reuses past experiences (sample efficiency)
     - Stabilizes training

3. **Target Network**:
   - Separate network for computing Q-targets
   - Updated periodically (every N steps) by copying main network
   - **Why target network?** Prevents moving target problem - Q-targets would change every step without it, making learning unstable

4. **Epsilon-Greedy Exploration**:
   ```
   ε(t) = ε_end + (ε_start - ε_end) * max(0, (ε_decay - t) / ε_decay)
   ```
   - Starts with ε=1.0 (pure exploration)
   - Decays to ε=0.05 (mostly exploitation)
   - **Why epsilon-greedy?** Simple, effective exploration strategy for discrete actions

5. **Dueling Architecture**:
   ```
   Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
   ```
   - Separates state value V(s) from action advantages A(s,a)
   - **Why dueling?** Can learn which states are valuable without learning action values for every action

**Hyperparameters**:
- Learning rate: 1e-3 (higher than PPO, common for Q-learning)
- Gamma: 0.99 (same as PPO)
- Epsilon start: 1.0 (full exploration)
- Epsilon end: 0.05 (minimal exploration)
- Epsilon decay: 20,000 steps
- Batch size: 128 (larger than PPO mini-batches)
- Replay capacity: 50,000

### 4. Neural Networks (`src/models/networks.py`)

**Architecture Choices**:

1. **CNN Backbone** (for image-like observations):
   - 3 convolutional layers with increasing channels (32, 64, 64)
   - ReLU activations
   - **Why CNN?** Efficiently processes spatial structure of the maze
   - **Why these sizes?** Standard architecture that works well for small grids

2. **MLP Backbone** (for flattened observations):
   - 2 hidden layers (256, 128 units)
   - ReLU activations
   - **Why MLP?** Simpler, faster for vector inputs (DQN uses flattened observations)

3. **Actor-Critic Head**:
   - Policy head: Linear layer → logits → softmax → action distribution
   - Value head: Linear layer → scalar value
   - **Why separate heads?** Policy and value have different objectives, separate heads allow specialized learning

4. **Dueling MLP**:
   - Shared feature extractor
   - Value stream: V(s)
   - Advantage stream: A(s,a) for each action
   - Combined: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
   - **Why subtract mean?** Ensures identifiability (V and A are not uniquely determined without this)

---

## Reinforcement Learning Algorithms

### PPO vs DQN: When to Use Which?

**PPO (Proximal Policy Optimization)**:
- **Best for**: On-policy learning, stable training, continuous improvement
- **Strengths**: 
  - Stable updates (clipping prevents large policy changes)
  - Good sample efficiency (uses data immediately)
  - Works well with continuous action spaces (though we use discrete here)
- **Trade-offs**: 
  - Requires on-policy data (can't reuse old experiences as effectively)
  - More complex (actor-critic architecture)

**DQN (Deep Q-Network)**:
- **Best for**: Off-policy learning, sample efficiency from replay
- **Strengths**:
  - Can learn from any past experience (off-policy)
  - Experience replay improves sample efficiency
  - Simpler conceptually (just Q-values)
- **Trade-offs**:
  - Requires discrete actions (we have this)
  - Can be unstable (mitigated by target network)
  - Exploration strategy (epsilon-greedy) is less sophisticated than entropy-based

**In Practice**: Both algorithms work well for maze navigation. PPO tends to be more stable, while DQN can be more sample-efficient with experience replay.

---

## Environment Design

### Observation Space

**4-Channel Grid Representation**:
```python
observation.shape = (H, W, 4)
```

- **Channel 0 (Agent)**: Binary mask showing agent position
- **Channel 1 (Walls)**: Binary mask showing wall locations
- **Channel 2 (Obstacles)**: Binary mask showing obstacle locations  
- **Channel 3 (Goal)**: Binary mask showing goal position

**Why this format?**
- CNNs excel at processing multi-channel spatial data
- Agent can see all relevant information simultaneously
- Separates different object types into channels (easier to learn)
- Standard format for vision-based RL

**Alternative (Flattened)**: For DQN, observations are flattened to 1D vectors. This is because DQN typically uses MLPs, though CNNs could work too.

### Reward Engineering

The reward structure is carefully designed to solve several problems:

1. **Sparse Rewards**: Only reaching the goal gives a large reward. Solution: Reward shaping (Manhattan distance) provides intermediate signals.

2. **Path Optimization**: Agent might find one path and stop improving. Solution: Efficiency bonus rewards shorter paths than previous best.

3. **Exploration vs Exploitation**: Agent might get stuck. Solution: Exploration bonus for new cells, repeat penalty for staying in place.

4. **Learning from Mistakes**: Terminating on collision prevents learning. Solution: Collisions are penalized but non-terminal.

### Episode Termination

- **Success**: Agent reaches goal → `terminated=True`
- **Timeout**: Max steps reached → `truncated=True`
- **Collision**: Agent hits wall/obstacle → **NOT terminated** (allows learning)

---

## User Interface

### Main Menu

**Color Palette**:
- Background: Deep navy gradient from RGB(10, 15, 25) to RGB(5, 8, 15)
- Primary accent: Indigo RGB(99, 102, 241) for selected items and highlights
- Secondary accent: Purple RGB(139, 92, 246) for badges and glows
- Text: White RGB(255, 255, 255) for primary text, gray RGB(156, 163, 175) for secondary
- Borders: Dark slate RGB(30, 41, 59)
- Cards: Semi-transparent dark RGB(20, 25, 35) with 220 alpha

**Interactive Elements**:
- Selected items highlighted in indigo with purple glow
- Hover states use lighter indigo RGB(129, 140, 248)
- Menu items use white text on dark background

### Visual Trainer

**Statistics Card Colors**:
- Success rate: Emerald green RGB(34, 197, 94)
- Average steps: Indigo RGB(99, 102, 241)
- Best path: Purple RGB(139, 92, 246)
- Current progress: Amber RGB(251, 191, 36)
- Exploration metric: Blue RGB(59, 130, 246)
- Return value: Pink RGB(236, 72, 153)

**Layout**: Cards arranged in 3-column grid with dark semi-transparent backgrounds and colored left accent bars.

### Maze Editor

**Element Colors**:
- Walls: Slate gray RGB(71, 85, 105)
- Obstacles: Red RGB(239, 68, 68)
- Start position: Green RGB(34, 197, 94)
- Goal position: Amber RGB(251, 191, 36)
- Grid lines: Dark slate RGB(30, 41, 59)
- Background: Navy gradient RGB(10, 15, 25) to RGB(5, 8, 15)
- Selected mode: Indigo RGB(99, 102, 241)

---

## Quickstart

```bash
# Setup
python -m venv .venv
. .venv/Scripts/activate   # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Launch application (recommended)
python scripts/main_menu.py

# Alternative: Direct script execution
python scripts/maze_editor.py          # Create/edit mazes
python scripts/visual_trainer.py --maze mazes/your_maze.json --algorithm ppo
python scripts/maze_tester.py --maze mazes/your_maze.json --algorithm dqn --mode train
```

---

## Usage Guide

### Main Menu Features

The main menu provides a centralized interface for all operations:

1. **Create New Maze**: Opens the maze editor to create a new maze from scratch
2. **Edit Existing Maze**: Select a maze from the list to edit it
3. **Test Algorithm on Maze**: Select a maze and algorithm to train/test

**Additional Features**:
- **Rename Maze**: Press `R` while a maze is selected to rename it
- **Delete Maze**: Press `Delete` while a maze is selected to delete it
- **Keyboard Navigation**: Use arrow keys to navigate, `Enter` to select, `Escape` to go back

### Creating a Maze

1. Launch main menu → "Create New Maze"
2. Use keyboard shortcuts to switch modes:
   - `W`: Wall mode
   - `O`: Obstacle mode
   - `S`: Start position
   - `G`: Goal position
   - `E`: Erase mode
3. Click/drag to place elements
4. Adjust grid size with `+`/`-` keys
5. Press `Enter` to save

### Editing an Existing Maze

1. Main menu → "Edit Existing Maze"
2. Select the maze you want to edit from the list
3. The maze will load with all its current elements
4. Make your changes
5. Press `Enter` to save (overwrites the original file)

### Testing Algorithms

1. Main menu → "Test Algorithm on Maze"
2. Select a maze from the list
3. Choose algorithm (PPO or DQN)
4. Watch real-time training with statistics dashboard

**Visual Trainer Features**:
- Real-time path visualization
- Live statistics dashboard
- Episode counter and success tracking
- Best path length tracking
- Exploration metric display (Entropy for PPO, Epsilon for DQN)
- Pause/Resume with `Space`
- Speed control with `+`/`-` keys

### Comparing Performance

Train both algorithms on the same maze and compare:
- Success rate (higher is better)
- Average steps (lower is better)
- Training stability (check learning curves)
- Path optimization (best path length)

---

## Technical Details

### Project Structure

```
AI-Project/
├── src/                         # Core library code
│   ├── envs/
│   │   └── maze_env.py          # Gymnasium-compatible environment
│   ├── agents/
│   │   ├── ppo.py               # PPO implementation
│   │   └── dqn.py               # DQN implementation
│   ├── models/
│   │   └── networks.py          # Neural network architectures
│   └── utils/
│       ├── rollout.py           # PPO experience buffer
│       ├── replay_buffer.py     # DQN experience buffer
│       └── logger.py            # Training logging
├── scripts/                     # Application scripts
│   ├── main_menu.py             # Main application menu (entry point)
│   ├── maze_editor.py          # Interactive maze creator/editor
│   ├── visual_trainer.py       # Real-time training visualization
│   ├── maze_tester.py          # Headless training/evaluation
│   ├── evaluate.py             # Agent evaluation script
│   ├── train_ppo.py            # PPO training script
│   └── train_dqn.py            # DQN training script
├── mazes/                       # Saved maze files (JSON)
├── runs/                        # Training outputs and checkpoints
│   ├── ppo/                    # PPO training results
│   └── dqn/                    # DQN training results
├── requirements.txt             # Python dependencies
├── pyrightconfig.json          # Type checking configuration
└── README.md                   # This file
```

### Maze File Format

```json
{
  "grid_size": 10,
  "walls": [[false, true, false, ...], ...],
  "obstacles": [[false, false, true, ...], ...],
  "start": [0, 0],
  "goal": [9, 9]
}
```

### Reproducibility

- All random number generators are seeded
- Checkpoints save model state and configuration
- Maze layouts saved as JSON for exact reproduction
- Training logs include hyperparameters and results

### Performance Considerations

- **Training Speed**: 
  - PPO updates more frequently (every 64 steps)
  - DQN updates every batch (when buffer has enough samples)
  - Visual trainer adds rendering overhead (~0.05s per step)
  - Use `maze_tester.py` for faster headless training
- **Memory**: 
  - DQN uses more memory (replay buffer: 50,000 samples)
  - PPO uses less (rollout buffer: 64 steps)
- **GPU Acceleration**: Automatically uses CUDA if available
- **Visualization**: 
  - Real-time rendering adds overhead
  - Use headless mode (`maze_tester.py`) for faster training
  - Visual trainer is best for demonstration and debugging
- **Quick Testing**: Use smaller step counts (1000-5000) for rapid iteration

---

## Advanced Features

### Stuck Detection & Recovery

The system includes intelligent stuck detection:
- Tracks recent positions (last 20 steps)
- Detects when agent visits ≤3 unique positions in threshold window
- Temporarily increases exploration (entropy/epsilon boost)
- Prevents infinite loops
- Automatically recovers from stuck states

### Path Optimization

After finding a successful path, the agent continues to optimize:
- Efficiency bonus rewards shorter paths than previous best
- Best path length tracked across episodes
- Agent incentivized to find optimal solution
- Visual feedback shows when new best path is found

### Adaptive Exploration

- **PPO**: 
  - Entropy coefficient starts high (0.3), decays to minimum (0.05) over 10,000 steps
  - Temporarily boosts to 0.5 when stuck detected
  - Adaptive decay based on training progress
- **DQN**: 
  - Epsilon starts at 1.0, decays to 0.05 over 20,000 steps
  - Linear decay schedule
  - Exploration handled internally by epsilon-greedy policy

### Maze Management

- **Save/Load**: Mazes saved as JSON files in `mazes/` directory
- **Rename**: Rename mazes directly from main menu (press `R`)
- **Delete**: Remove mazes from main menu (press `Delete`)
- **Edit**: Load and modify existing mazes
- **Grid Size**: Adjustable during creation/editing

---

## Conclusion

This platform demonstrates enterprise-grade software engineering principles:
- **Modularity**: Clean separation of concerns
- **Extensibility**: Easy to add new algorithms or features
- **Usability**: Intuitive interfaces with real-time feedback
- **Performance**: Efficient implementations with hardware acceleration
- **Documentation**: Comprehensive explanations of design decisions

The combination of sophisticated RL algorithms, well-designed environment, and professional UI makes this a powerful tool for both research and education in reinforcement learning.

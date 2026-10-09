![CI](https://github.com/emiliomunozai/mountain_car/actions/workflows/ci.yml/badge.svg?branch=main)

# MountainCar Reinforcement Learning

This project implements and compares two value-based reinforcement learning agents for Gymnasium's `MountainCar-v0` environment:

- tabular Q-Learning over a discretized state space; and
- a Deep Q-Network (DQN) that approximates the action-value function from the continuous observation.

The project includes training and evaluation code, a command-line interface, model persistence, and an experimental analysis of exploration in a flat-reward environment. The agents are implemented directly with NumPy and PyTorch rather than with a high-level reinforcement learning library.

## 1. MountainCar-v0 environment

MountainCar places an under-powered car in a valley. The car cannot drive straight up the right-hand hill; it must move back and forth to build momentum and reach the flag at position `0.5`.

### State

The observation is a continuous vector with two values:

| Index | Variable | Range |
|---:|---|---:|
| 0 | Position | `-1.2` to `0.6` |
| 1 | Velocity | `-0.07` to `0.07` |

### Actions

| Action | Meaning |
|---:|---|
| 0 | Push left |
| 1 | Do not push |
| 2 | Push right |

### Reward and termination

The environment gives a reward of `-1` at every step. An episode terminates when the car reaches the flag and is truncated after 200 steps. There is no additional success reward. Consequently, for a successful episode:

```text
episode return = -number of steps
```

Less-negative returns are better: they represent reaching the flag in fewer steps. An episode that times out has return `-200`.

## 2. Installation

The project uses `uv` for environment and dependency management:

```bash
uv sync
```

## 3. CLI usage

All commands are exposed through the `mountaincar` executable:

```bash
uv run mountaincar <command>
```

| Command | Purpose |
|---|---|
| `version` | Show the package version |
| `list` | List agents and save-file status |
| `inspect` | Inspect spaces and sample transitions |
| `init <agent>` | Initialize and save an untrained agent |
| `train <agent>` | Train an agent, resuming its save when present |
| `load <agent>` | Show saved-agent information and optionally evaluate it |
| `sim <agent>` | Simulate episodes with step-by-step text output |
| `render <agent>` | Render episodes in a graphical window |
| `delete <agent>` | Delete an agent save file |

The available agent names are `qlearning` and `dqn`.

Examples:

```bash
uv run mountaincar inspect --steps 3
uv run mountaincar train qlearning --episodes 50000
uv run mountaincar load qlearning --eval
uv run mountaincar train dqn --episodes 3000
uv run mountaincar load dqn --eval
uv run mountaincar render dqn --episodes 3
```

Saved models are written to `saves/`. Existing saved models should be evaluated with the same environment and code configuration used to produce them.

## 4. Q-Learning implementation

The Q-Learning agent converts the two continuous observation dimensions into a `20 x 20` grid. Each pair of bin indices is a discrete state, giving a maximum of 400 possible states. The Q-table stores three action values for each state.

For a transition `(s, a, r, s')`, the implementation uses the tabular Q-Learning update:

```text
Q(s,a) <- Q(s,a) + alpha * [r + gamma * max_a' Q(s',a') - Q(s,a)]
```

For a genuinely terminal transition, the bootstrap term is omitted. Time-limit truncation is kept distinct from environment termination, so the agent can continue bootstrapping through the step limit.

### Q-Learning configuration

| Hyperparameter | Value |
|---|---:|
| Bins per dimension | 20 |
| Maximum discrete states | 400 |
| Learning rate | 0.1 |
| Discount gamma | 0.99 |
| Initial epsilon | 1.0 |
| Minimum epsilon | 0.01 |
| Epsilon decay | 0.9995 |

Action selection is epsilon-greedy during training. Evaluation and rendering use `deterministic=True`, which disables exploration and selects the greedy action.

## 5. DQN implementation

The DQN receives the raw continuous state and predicts one Q-value per action. Its network is:

```text
2 inputs -> Linear(128) -> ReLU -> Linear(128) -> ReLU -> Linear(3)
```

There is no output activation because the outputs are Q-values, not probabilities. The network contains 17,283 trainable parameters.

Training uses:

- a replay buffer with capacity 100,000 transitions;
- mini-batches of 64 transitions sampled from the buffer;
- an online network for current Q-values;
- a target network for the Bellman target;
- target-network synchronization every 10 episodes; and
- the Adam optimizer with mean-squared Bellman error.

For non-terminal transitions, the target is `r + gamma * max Q_target(s', a')`; for terminal transitions, it is simply `r`. The training loop stores `terminated`, rather than `terminated or truncated`, for this purpose.

### DQN configuration

| Hyperparameter | Value |
|---|---:|
| Learning rate | 0.001 |
| Discount gamma | 0.99 |
| Initial epsilon | 1.0 |
| Minimum epsilon | 0.01 |
| Epsilon decay | 0.995 |
| Batch size | 64 |
| Replay capacity | 100,000 |
| Target synchronization | Every 10 episodes |
| Exploratory action duration | 20 steps |
| Device in final evaluation | CPU |

## 6. Exploration problem and diagnosis

The first DQN experiment used standard independent per-step epsilon-greedy exploration. After 1,000 MountainCar episodes, the reward remained flat at `-200`; the replay buffer had reached 100,000 transitions and epsilon had reached `0.01`, but the agent had not reached the flag.

Two diagnostic experiments clarified the cause:

1. **CartPole validation.** The same DQN implementation was trained on `CartPole-v1` for 200 episodes with `epsilon_decay=0.98`. Average rewards increased substantially, including 23.08 at episode 25, 159.92 at episode 75, 235.68 at episode 100, and 268.08 at episode 200. This supported the conclusion that the network, replay buffer, Bellman update, optimizer, and target network were capable of learning.
2. **Random MountainCar policy.** Independent random actions reached the flag in `0/300` episodes. Random per-step actions do not reliably produce the sustained directional runs needed to build momentum.

Exploration was changed so that, when an exploratory action is selected, it is held for 20 steps. This temporally correlated exploration makes sustained motion possible without changing the reward, network, or Bellman learning rule.

| Training point | Average reward |
|---|---:|
| Episode 10/20 | -198.80 |
| Episode 20/20 | -190.50 |

The initial Q-Learning evaluation also exposed an evaluation-only issue. The deterministic action-selection condition was corrected from the inverted form `if deterministic or ...` to the intended `if not deterministic and ...`. The training logic was functioning; therefore, 50,000 episodes is reported as the final evaluated checkpoint, not as the exact episode at which learning first became successful.

## 7. Experimental results

The following results are the verified final evaluations. Each evaluation used 10 episodes with greedy action selection.

### Q-Learning

| Metric | Result |
|---|---:|
| Final evaluated checkpoint | 50,000 episodes |
| States visited | 298 / 400 |
| Final epsilon | 0.0100 |
| Mean reward | -153.80 |
| Standard deviation | 8.62 |
| Episodes reaching the flag | 10 / 10 |

### DQN

| Metric | Result |
|---|---:|
| Final checkpoint | 3,000 episodes |
| Final epsilon | 0.0100 |
| Mean reward | -109.20 |
| Standard deviation | 4.71 |
| Episodes reaching the flag | 10 / 10 |
| Device | CPU |

For reference, the DQN also reached mean reward `-115.80` with standard deviation `11.07` after 2,500 episodes, with success in 10/10 evaluation episodes. The 3,000-episode checkpoint is used for the final comparison.

## 8. Q-Learning versus DQN

Because MountainCar gives `-1` per step, the absolute reward corresponds directly to episode length for successful episodes.

| Criterion | Q-Learning | DQN |
|---|---|---|
| Final checkpoint | 50,000 episodes | 3,000 episodes |
| Mean reward / approximate steps | -153.80 / 153.8 | -109.20 / 109.2 |
| Standard deviation | 8.62 | 4.71 |
| Success | 10/10 | 10/10 |
| State representation | 20 x 20 discretized grid | Continuous state with neural approximation |
| Representation size | 298 of 400 states visited | 17,283 trainable parameters |

The DQN reached the goal approximately 44.6 steps earlier on average:

```text
153.8 - 109.2 = 44.6 steps
```

Relative to Q-Learning, this is approximately a 29% reduction in average episode length:

```text
44.6 / 153.8 ≈ 29%
```

### Interpretation

- **Stability:** DQN had the lower evaluation standard deviation (4.71 versus 8.62), indicating more consistent episode lengths in this evaluation. This remains a 10-episode measurement.
- **Learning speed:** The final DQN checkpoint used 3,000 training episodes, whereas the final evaluated Q-Learning checkpoint used 50,000. This compares the documented experiments; it is not a universal sample-efficiency claim.
- **Final performance:** Both agents reached the flag in all 10 evaluation episodes. DQN reached it in fewer steps on average.
- **Interpretability:** The Q-table is compact and directly inspectable, making the tabular policy easier to explain.
- **Discretization limitations:** Q-Learning loses information when nearby continuous states share a bin, and finer grids increase table size and data requirements. Its result is tied to the selected 20-bin representation.
- **Generalization:** DQN uses the continuous observation directly and can interpolate between observed states. This gives it greater representational flexibility, at the cost of more parameters and less transparent decisions.
- **Implementation complexity:** Q-Learning requires a discretizer, a table, and a TD update. DQN adds a neural network, replay sampling, tensor updates, optimization, and a target network.
- **Exploration:** MountainCar's flat and sparse success signal makes independent random actions ineffective; temporally correlated exploration was essential for the successful DQN experiment.

## 9. Conclusions

Both implementations solved MountainCar under their final evaluation conditions. Q-Learning provides a simple, interpretable baseline for this low-dimensional problem. DQN required more machinery and a targeted exploration strategy, but its continuous-state representation produced the stronger result in these experiments: the same 10/10 success rate with approximately 29% fewer steps per episode and lower measured variability.

The experiments also show that a flat reward curve does not automatically mean that the DQN update is incorrect. In this environment, the data-collection policy must first generate trajectories that contain the behavior needed to reach the flag.

## 10. Evidence and manually created diagrams

The assignment requires the diagrams and screenshots to be produced manually. They are intentionally not generated or fabricated in this repository. Add the final artifacts here when ready, for example:

- `[Manual diagram: MountainCar state/action/reward flow]`
- `[Manual diagram: Q-Learning discretization and TD update]`
- `[Manual diagram: DQN online network, replay buffer, and target network]`
- `[Manual screenshot: Q-Learning training/evaluation output]`
- `[Manual screenshot: DQN diagnosis and final evaluation output]`

The numerical results in this README are the verified experimental values provided for the final report. The placeholders above are references for student-created visual evidence and are not claims that those files already exist.

## 11. Project structure

```text
src/mountain_car/
├── cli.py              # command-line interface and evaluation utilities
└── agents/
    ├── qlearning.py    # tabular Q-Learning agent
    └── dqn.py          # QNetwork, ReplayBuffer, and DQNAgent
saves/                  # local agent save files
EXERCISES.md            # original exercise prompts and diagnostic guidance
pyproject.toml          # package metadata and dependencies
uv.lock                # locked dependency versions
```

`EXERCISES.md` is retained as the original learning record and supporting material. This README documents the completed implementation and experiments.

# Thesis presentation

<https://campuscvut-my.sharepoint.com/:p:/g/personal/mrkosmic_cvut_cz/IQDbQSLeaUP_SI8s4acbHOsbAU8uEZ0k6doH0hlrZffHd_M?e=VHJEO0>

- videos
  - <https://youtu.be/fa1zT2-eddA?si=sku8gPYBZiDHiT8I>

- RL should be a local planner
  - so trying to steer around obstacles, keeping good shape for next move (not solving the problems with  nonconvexity, etc)

- should know about the cable shape
  - so control is not reactive (THIS FAILED MISERABLY)

- randomly select start and end on the map
  - maybe this could be a problem

- Reward
  - reducing distance to the goal
  - shaping rewards for hitting obstacle
  - high velocity
  -  

# Code

<https://github.com/majklost/RRT-RL-2D>

## Approaches

only thing that worked a bit well - Radius RL
observations

All experiments use **PPO**. Roughly 14 env variants in 5 families:

| Family | File | Idea |
|---|---|---|
| **CableEnv** | `envs/cable_env.py` | Per-segment goal vectors, agent outputs 20-D forces. Variants: `Naive` (absolute positions), `InnerAngles` (sin/cos of joints), `BigTest` (engineered features), `PID` (velocity-target wrapper). |
| **DodgeEnv** | `envs/dodge_env.py` | Hardcoded attractive force toward goal; RL only learns a *steering correction*: `force = 2·rl_action + advised_action`. Obs is only obstacle vectors + normalized step counter. Variants: `Vel`, `Penalty`, `Reduction`, `PenaltyReduction`. |
| **BlendEnv** | `envs/blend_env.py` | Reduces action space: RL outputs **one blend coefficient per segment** in [-1,1] that mixes attractive vs. repulsive force. `BlendStrength` adds magnitude. |
| **CableRadius** | `envs/cable_radius.py` | Single-point goal with a radius threshold (not per-segment). Adds nearest-obstacle observations and optionally velocities. |
| **LastEnv** | `envs/last_env.py` | Most elaborate attempt: ~60-D obs (positions, segment vectors, sin/cos angles, goal vectors+distances, obstacle vectors+distances), curriculum across 4 maps (Empty → AlmostEmpty → StandardStones → Piped), 8M steps each, net [512,256,256]. |

**Defaults**: `pi=[256,256], vf=[256,256]`, Tanh, 32–128 vectorised envs, 4M–32M timesteps, `TimeLimit=1000` per episode, VecNormalize on obs/reward. Optuna search space defined in `rrt_rl_2D/RL/hyperparams.py` but only a handful of tuned runs actually executed.

---

### Types of goals

- Bezier goal
  - create a  non self-intersecting shape of cable
    - so cable should match even the shape
- 2D goal
  - sufficient if whole cable is near the goal

## Demo showcase

`python scripts/controllable_tests/asset_test.py`

- manually by arrows move the agent (only per segment), tab and shift tab to cycle betweeen segments

```bash
pip install -e .
python scripts/init_RL.py            # creates ./experiments/RL
python scripts/RL/cable-standard.py  # or cable-blend.py, cable-radius.py, cable-last.py
tensorboard --logdir experiments/RL/
```

after learning

```
python ./scripts/RL/play_experiment.py cable-standard-standard  
```

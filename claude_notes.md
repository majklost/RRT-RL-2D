# RRT-RL-2D — RL Analysis Notes

Notes prepared to explain the RL portion of the thesis to a lecturer: what the project does, what was tried, why training likely failed, and how to launch a new training run.

---

## 1. What the project does

A **2D PyMunk simulation** of a deformable **cable** (10 rigid segments linked by spring constraints, total length ~300 units) that must be steered from a start to a goal through an obstacle field. The broader thesis combines **RRT global planning** with **RL local control** — RRT proposes waypoints, RL drives the cable between them while dodging obstacles.

- Environments: `rrt_rl_2D/envs/` (Gymnasium API)
- Physics config: `rrt_rl_2D/simulator/standard_config.py`
  (map 1900×800, damping 0.15, gravity 0, fps 60, 10 segments, 300-unit cable, max 1000 steps)
- RL library: **Stable-Baselines3 PPO** (PyTorch backend)
- Training entry points: `scripts/RL/cable-*.py`

---

## 2. Approaches tried

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

## 3. Why RL did not learn properly

1. **Sparse + miscalibrated reward** (`cable_env.py:79-100`): per-step `reward = 10·Δpotential − 20`. Moving 1 px toward goal yields −10. Collision: −1000; 1000-step timeout doing nothing: −20000. Agent learns *not to try*.
2. **First-step zero-reward bug** (`cable_env.py:95-96`, `dodge_env.py:38-42`): `last_target_potential` initialised to 0, so first step's potential difference is overwritten with 0 — no learning signal on step 1.
3. **Observations don't include cable state** in baseline CableEnv — only goal-relative vectors. Agent cannot perceive its own shape, velocity, or which segment is near an obstacle, yet must control 10 segments individually.
4. **Hidden baseline action breaks credit assignment** (`dodge_env.py:48-67`): applied force is `2·rl + advised`, but `advised` is not in the observation. Agent cannot tell whether the goal was reached because of *its* output or the hidden attractor.
5. **Action/observation dimensional mismatch**: DodgeEnv obs ≈ 21 but action = 20-D forces; agent cannot condition each segment force on local state it doesn't observe.
6. **Random start *and* goal every episode** (`rrt_env.py:169-172`, also noted in `Notes.md`): every episode is a different task, forcing PPO to learn a universal controller rather than a single skill.
7. **`TimeLimit=1000` too short** relative to cable speed (~5 units/step) vs. map size (1900×800). Barely enough to reach the goal with a perfect policy; almost no exploration budget.
8. **Linear blend assumption in BlendEnv** ignores coupled cable dynamics — optimal force is not a convex combination of attractor and repulsor.
9. **Angle-straightness bonus in LastEnv** (`last_env.py:85-88`, `+0.1·Σcos(angles)`) creates a local optimum: stay still and straight.
10. **No systematic hyperparameter search** was actually executed despite the Optuna scaffolding.

---

## 4. How to train a new agent

```bash
pip install -e .
python scripts/init_RL.py            # creates ./experiments/RL
python scripts/RL/cable-standard.py  # or cable-blend.py, cable-radius.py, cable-last.py
tensorboard --logdir experiments/RL/
```

To make a new experiment, copy any function in `scripts/RL/cable-*.py` and change:

- the **maker** (`StandardCableMaker`, `BlendMaker`, `DodgeEnvMaker`, `LastEnvMaker`, … from `rrt_rl_2D/CLIENT/makers/`),
- the **map name** (`'Empty' | 'AlmostEmpty' | 'StandardStones' | 'NonConvex' | 'Piped'`),
- `STANDARD_CONFIG` fields (`seg_num`, `cable_length`, `threshold`, `damping`),
- PPO hyperparameters and `net_arch`,
- `total_timesteps`.

To change reward/observations, edit or subclass the env in `rrt_rl_2D/envs/*.py`:
`_get_observation`, `_get_reward`, `_create_observation_space`, `_create_action_space`.

Per-run outputs (model checkpoints, VecNormalize stats, TensorBoard logs, metadata) are saved under `experiments/RL/<run_name>/`.

### Recommended starting point for the lecturer
`BlendEnv` on `AlmostEmpty` with:

- a **fixed** start/goal (disable `resetable=True`),
- shorter episodes (~300 steps),
- denser reward (drop the −20 per-step constant; scale terminal bonuses to ±100).

This isolates whether PPO can learn anything at all in this physics before tackling randomised tasks.

---

## 5. Key file references

| Aspect | File | Lines |
|---|---|---|
| Base RL env | `rrt_rl_2D/envs/rrt_env.py` | 17-173 |
| CableEnv (reward, obs) | `rrt_rl_2D/envs/cable_env.py` | 9-283 |
| DodgeEnv (blended action) | `rrt_rl_2D/envs/dodge_env.py` | 9-197 |
| BlendEnv | `rrt_rl_2D/envs/blend_env.py` | 7-102 |
| CableRadius | `rrt_rl_2D/envs/cable_radius.py` | 12-121 |
| LastEnv (richest obs) | `rrt_rl_2D/envs/last_env.py` | 8-92 |
| Env makers | `rrt_rl_2D/CLIENT/makers/makers.py` | 34-498 |
| Hyperparam search space | `rrt_rl_2D/RL/hyperparams.py` | 5-54 |
| Training utilities | `rrt_rl_2D/RL/training_utils.py` | 14-166 |
| Physics config defaults | `rrt_rl_2D/simulator/standard_config.py` | 4-36 |
| Training scripts | `scripts/RL/cable-*.py` | — |

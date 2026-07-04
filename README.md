# drone_sim — RL quadrotor simulator for Apple silicon

A self-contained physics simulator + PPO trainer for autonomous quadrotors,
built to run fast on a MacBook (M-series). No MuJoCo, no Isaac Gym — just
PyTorch. Simulates ~150,000 physics steps/second on an M3 CPU with 1024
drones in parallel, enough to train a hover policy from scratch in a few
minutes.

![A trained quadrotor approaching and holding its target](assets/hover_demo.gif)

*A policy trained from scratch in ~2 minutes: it flies to the red target and
holds within 2 cm, correcting against randomized mass, thrust, and drag.*

## Quick start

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python torch numpy matplotlib

# Train a hover policy (~2-3 min, ~20M sim steps)
.venv/bin/python train.py --task hover

# Watch it fly LIVE in a window (real-time 3D, flies autonomously)
.venv/bin/python watch.py --task hover

# Or render to files: runs/hover/trajectory.png and runs/hover/flight.gif
.venv/bin/python evaluate.py --task hover

# Harder task: fly through a sequence of random waypoints
.venv/bin/python train.py --task waypoint --updates 600
.venv/bin/python watch.py --task waypoint          # steer it yourself, live
```

## Flying it yourself — manual waypoints

`watch.py` opens a live 3D window and, for the waypoint task, lets you drop new
waypoints by hand while the drone is airborne. Focus the window and press:

| Key | Action |
|-----|--------|
| `w` / `s` | waypoint forward / back (+y / −y) |
| `a` / `d` | waypoint left / right (−x / +x) |
| `r` / `f` | waypoint up / down |
| `space` | drop a random waypoint |
| `c` | clear the queue and hover in place |

Each key drops a waypoint 1.5 m from the drone's current position; it flies your
queued points in order, then holds at the last one. The first keypress switches
the env out of random-target mode into your manual queue.

**From your own code**, it's a one-liner — the same API the viewer uses:

```python
from drone_sim.env import QuadrotorEnv
env = QuadrotorEnv(num_envs=1, task="waypoint")
env.reset()
env.queue_waypoint([2.0, 1.0, 1.5])   # fly here first
env.queue_waypoint([-2.0, 0.0, 2.0])  # then here
# ... step the env; the drone visits them in order.
```

The queue lives in `QuadrotorEnv` (`drone_sim/env.py`): `queue_waypoint()` and
the `manual_queue` / `manual_mode` fields. That's the place to script flight
paths, patrol loops, or read waypoints from a file.

## What's in the box

```
drone_sim/
  physics.py   Batched rigid-body quadrotor dynamics (pure PyTorch)
  env.py       Vectorized RL environment: hover + waypoint tasks
  ppo.py       PPO with GAE, tuned for the batched env
  viz.py       Training curves, 3D trajectory plots, flight GIFs
train.py       Training CLI
watch.py       Live real-time 3D viewer (opens a window, flies forever)
evaluate.py    Roll out a trained policy, render plots + GIF
```

## Physics model

- X-configuration quadrotor: 0.75 kg, 25 cm motor-to-motor, thrust-to-weight 2.75
- Rigid-body dynamics with quaternion attitude and Euler's rotation equation
  (diagonal inertia), semi-implicit Euler integration at 250 Hz with 5
  substeps per 50 Hz control step
- First-order motor lag (τ = 33 ms) — the policy has to plan around actuator
  delay, like a real drone
- Rotor yaw drag torque, linear body drag, angular damping
- **Domain randomization** per episode: mass ±20%, thrust gain ±10%,
  drag ±30% — the standard sim-to-real trick so policies don't overfit to
  one airframe

All state is `[num_envs, ...]` tensors, so thousands of drones step in one
vectorized call.

## RL setup

- **Observation (22-d):** vector to target, velocity, body→world rotation
  matrix, angular velocity, previous action
- **Action (4-d):** normalized per-rotor thrust commands in [-1, 1]
- **Reward:** shaped — proximity to target (exponential), upright bonus,
  alive bonus, penalties on speed, spin, and action jerk; big penalty on
  crash/flyaway; waypoint task adds a capture bonus
- **Algorithm:** PPO, 1024 parallel envs × 64-step rollouts (65k samples per
  update), GAE(λ=0.95), clip 0.2, correct time-limit bootstrapping

## Device notes (M-series Macs)

`--device auto` picks **CPU**, on purpose. MPS (the Metal GPU backend) works
(`--device mps`) but is ~5× slower for this workload: the physics loop is
many small kernel launches and GPU dispatch overhead dominates. CPU on an M3
does ~150k steps/s. MPS would win with much larger networks or 10k+ envs.

## Extending it

- **New tasks:** subclass or edit `env.py` — velocity tracking, trajectory
  following, gates/racing are all a `_sample_targets` + `_reward` change.
- **Sensors:** the obs is assembled in `QuadrotorEnv._obs`; add noise or an
  IMU-only variant there for realism.
- **Sim-to-real:** widen the randomization ranges in `QuadrotorParams`, add
  observation noise and action latency, then export the actor MLP (2×128
  tanh — tiny) to run onboard.

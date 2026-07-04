"""Roll out a trained policy and save a 3D trajectory plot + animated GIF.

Example:
  python evaluate.py --task hover
  python evaluate.py --task waypoint --seconds 12
"""

import argparse
import os

import numpy as np
import torch

from drone_sim.env import QuadrotorEnv
from drone_sim.ppo import PPOTrainer
from drone_sim.viz import animate_flight, plot_trajectory


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["hover", "waypoint", "recovery"], default="hover")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="runs")
    args = ap.parse_args()

    ckpt = args.ckpt or os.path.join(args.out, args.task, "policy.pt")
    run_dir = os.path.dirname(ckpt)

    # Single env on CPU is plenty for evaluation.
    env = QuadrotorEnv(num_envs=1, task=args.task, device="cpu",
                       episode_seconds=args.seconds, seed=args.seed)
    trainer = PPOTrainer(env, device="cpu")
    trainer.load(ckpt)
    net = trainer.net

    obs = env.reset()
    positions, targets, ups = [], [], []
    steps = int(args.seconds / env.sim.p.control_dt)
    dt = env.sim.p.control_dt
    crashed = False
    hits = 0
    recover_time = None            # seconds to first reach a stable upright state
    for i in range(steps):
        positions.append(env.sim.pos[0].numpy().copy())
        targets.append(env.target[0].numpy().copy())
        ups.append(env.sim.up_vector[0].numpy().copy())
        action, _, _ = net.act(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        # Read from info (captured before the env's auto-reset) so a crash
        # doesn't zero the count we report.
        hits = int(info["waypoints_hit"][0])
        # First moment the drone holds upright+calm for ~0.3 s counts as recovered.
        if recover_time is None and float(info["upright_streak"][0]) >= 15:
            recover_time = (i + 1) * dt
        if terminated[0]:
            crashed = True
            break

    positions = np.array(positions)
    targets = np.array(targets)
    ups = np.array(ups)

    final_dist = np.linalg.norm(targets[-1] - positions[-1])
    print(f"flew {len(positions) * env.sim.p.control_dt:.1f}s, "
          f"crashed={crashed}, final distance to target={final_dist:.2f} m")
    if args.task == "waypoint":
        print(f"waypoints hit: {hits}")
    if args.task == "recovery":
        if recover_time is not None:
            print(f"RECOVERED in {recover_time:.2f} s (righted itself and stabilized)")
        else:
            print("did not stabilize within the episode")

    png = os.path.join(run_dir, "trajectory.png")
    gif = os.path.join(run_dir, "flight.gif")
    plot_trajectory(positions, targets, png)
    animate_flight(positions, targets, ups, gif, control_dt=env.sim.p.control_dt)
    print(f"saved {png}\nsaved {gif}")


if __name__ == "__main__":
    main()

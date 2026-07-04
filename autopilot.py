"""Autopilot: fly a fixed course of checkpoints, on repeat, autonomously.

This is the "teach me" example. Everything you'd want to change is at the top:
edit the CHECKPOINTS list, save, and re-run. The drone flies them in order,
loops back to the start, and keeps going forever — no keyboard needed.

Run it:
  python autopilot.py                 # live 3D window
  python autopilot.py --no-window      # headless, just prints progress

How it works: each [x, y, z] below is a point in metres (z is altitude/height).
The trained waypoint policy already knows how to *fly* to a target; this script
just feeds it the next checkpoint each time it reaches the current one.
"""

# ---------------------------------------------------------------------------
# YOUR FLIGHT COURSE — edit these. Each row is [x, y, z] in metres.
# The drone visits them top-to-bottom, then loops back to the first one.
# x/y are left-right/forward-back; z is height above the ground.
# ---------------------------------------------------------------------------
CHECKPOINTS = [
    [ 2.0,  2.0, 1.5],   # far corner, low
    [-2.0,  2.0, 2.5],   # other corner, high
    [-2.0, -2.0, 1.0],
    [ 2.0, -2.0, 2.5],
    [ 0.0,  0.0, 2.0],   # back through the middle
]
# ---------------------------------------------------------------------------

import argparse
import os

import numpy as np
import torch

from drone_sim.env import QuadrotorEnv
from drone_sim.ppo import PPOTrainer
from drone_sim.physics import quat_to_rotmat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join("runs", "waypoint", "policy.pt"))
    ap.add_argument("--no-window", action="store_true", help="run headless")
    ap.add_argument("--speed", type=float, default=1.0, help="1.0 = real time")
    args = ap.parse_args()

    if not os.path.exists(args.ckpt):
        raise SystemExit(f"No trained policy at {args.ckpt}.\n"
                         f"Train one first:  python train.py --task waypoint")

    # One drone, waypoint task, effectively endless episode.
    env = QuadrotorEnv(num_envs=1, task="waypoint", device="cpu",
                       episode_seconds=1e6)
    trainer = PPOTrainer(env, device="cpu")
    trainer.load(args.ckpt)
    net = trainer.net
    obs = env.reset()
    dt = env.sim.p.control_dt

    # Load the course into the drone's waypoint queue.
    for cp in CHECKPOINTS:
        env.queue_waypoint(cp)

    # Optional live window.
    ax = None
    if not args.no_window:
        import matplotlib
        for backend in ("macosx", "TkAgg", "QtAgg"):
            try:
                matplotlib.use(backend, force=True); break
            except Exception:
                continue
        import matplotlib.pyplot as plt
        plt.ion()
        fig = plt.figure(figsize=(8, 7))
        ax = fig.add_subplot(projection="3d")

    trail = []
    laps, hits_prev = 0, 0
    step = 0
    while True:
        action, _, _ = net.act(obs, deterministic=True)
        obs, _, terminated, _, info = env.step(action)

        # Keep the course looping: whenever the queue runs low, add it again.
        if len(env.manual_queue) < len(CHECKPOINTS):
            for cp in CHECKPOINTS:
                env.queue_waypoint(cp)

        hits = int(env.waypoints_hit[0])
        if hits // len(CHECKPOINTS) > laps:
            laps = hits // len(CHECKPOINTS)
            print(f"completed lap {laps}  ({hits} checkpoints total)")

        if terminated[0]:
            # Rare: destabilized under domain randomization. Reset and reload.
            print("drone destabilized — resetting and resuming course")
            obs = env.reset()
            for cp in CHECKPOINTS:
                env.queue_waypoint(cp)
            trail.clear()

        if ax is not None and step % 2 == 0:
            import matplotlib.pyplot as plt
            pos = env.sim.pos[0].numpy()
            tgt = env.target[0].numpy()
            R = quat_to_rotmat(env.sim.quat)[0].numpy()
            trail.append(pos.copy())
            if len(trail) > 150:
                trail.pop(0)
            tr = np.array(trail)

            ax.clear()
            # Draw the whole planned course as faint markers.
            course = np.array(CHECKPOINTS)
            ax.scatter(course[:, 0], course[:, 1], course[:, 2],
                       color="gray", marker="o", s=30, alpha=0.4)
            ax.plot(tr[:, 0], tr[:, 1], tr[:, 2], color="tab:blue", alpha=0.5)
            up = pos + 0.35 * R[:, 2]
            ax.plot([pos[0], up[0]], [pos[1], up[1]], [pos[2], up[2]],
                    color="tab:orange", lw=2)
            ax.scatter([pos[0]], [pos[1]], [pos[2]], color="black", s=50)
            ax.scatter([tgt[0]], [tgt[1]], [tgt[2]], color="red", marker="*", s=200)
            span = 3.5
            ax.set_xlim(pos[0]-span, pos[0]+span)
            ax.set_ylim(pos[1]-span, pos[1]+span)
            ax.set_zlim(max(0, pos[2]-span), pos[2]+span)
            ax.set_title(f"autopilot  |  lap {laps}  |  checkpoints hit: {hits}")
            plt.pause(max(1e-3, 2 * dt / args.speed))
            if not plt.fignum_exists(fig.number):
                break

        step += 1


if __name__ == "__main__":
    main()

"""Live 3D viewer: watch a trained policy fly a drone in real time.

Opens a window and runs the policy forever (or for --seconds), rendering the
drone as it flies. The drone chases waypoints/target autonomously; close the
window to stop.

Examples:
  python watch.py --task waypoint      # fly through endless random waypoints
  python watch.py --task hover         # approach and hold a point
  python watch.py --task waypoint --speed 0.5   # half real-time (slow-mo)
"""

import argparse
import os

import matplotlib

for _backend in ("macosx", "TkAgg", "QtAgg"):
    try:
        matplotlib.use(_backend, force=True)
        break
    except Exception:
        continue
import matplotlib.pyplot as plt
import numpy as np
import torch

from drone_sim.env import QuadrotorEnv
from drone_sim.ppo import PPOTrainer


def rotor_world_positions(pos, rotmat, arm=0.16):
    """Four rotor tip positions in world frame for drawing the airframe."""
    L = arm / (2.0 ** 0.5)
    body = np.array([[L, -L, 0], [-L, -L, 0], [-L, L, 0], [L, L, 0]])
    return pos + body @ rotmat.T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["hover", "waypoint"], default="waypoint")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--seconds", type=float, default=0.0, help="0 = run forever")
    ap.add_argument("--speed", type=float, default=1.0, help="1.0 = real time")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ckpt = args.ckpt or os.path.join("runs", args.task, "policy.pt")
    if not os.path.exists(ckpt):
        raise SystemExit(f"no policy at {ckpt} — train one first: "
                         f"python train.py --task {args.task}")

    # Long episodes so a hover drone doesn't reset every few seconds.
    env = QuadrotorEnv(num_envs=1, task=args.task, device="cpu",
                       episode_seconds=1e6, seed=args.seed)
    trainer = PPOTrainer(env, device="cpu")
    trainer.load(ckpt)
    net = trainer.net
    obs = env.reset()
    dt = env.sim.p.control_dt

    plt.ion()
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(projection="3d")

    trail_len = 120
    trail = []

    # ---- Manual waypoint controls -------------------------------------------
    # Keys drop a new waypoint relative to where the drone is right now, so you
    # steer it around by hand. First keypress switches the env into manual mode
    # (no more random targets); the drone flies your queue in order.
    STEP = 1.5  # metres per keypress

    def on_key(event):
        pos = env.sim.pos[0].clone()
        offsets = {
            "w": (0, STEP, 0), "s": (0, -STEP, 0),      # forward / back (+y/-y)
            "a": (-STEP, 0, 0), "d": (STEP, 0, 0),      # left / right (-x/+x)
            "r": (0, 0, STEP), "f": (0, 0, -STEP),      # up / down
        }
        if event.key in offsets:
            wp = pos + torch.tensor(offsets[event.key])
            wp[2] = wp[2].clamp(min=0.4)
            env.queue_waypoint(wp)
        elif event.key == " ":                          # random waypoint
            env.queue_waypoint(env._sample_targets(1)[0])
        elif event.key == "c":                          # clear queue, hover here
            env.manual_queue.clear()
            env.target[0] = env.sim.pos[0]

    fig.canvas.mpl_connect("key_press_event", on_key)

    def draw():
        ax.clear()
        pos = env.sim.pos[0].numpy()
        tgt = env.target[0].numpy()
        rot = env.sim.up_vector  # not used directly; get full matrix below
        from drone_sim.physics import quat_to_rotmat
        R = quat_to_rotmat(env.sim.quat)[0].numpy()

        trail.append(pos.copy())
        if len(trail) > trail_len:
            trail.pop(0)
        tr = np.array(trail)

        # Airframe: rotor arms as an X, plus an up arrow for orientation.
        rotors = rotor_world_positions(pos, R)
        ax.plot([rotors[0, 0], rotors[1, 0]], [rotors[0, 1], rotors[1, 1]],
                [rotors[0, 2], rotors[1, 2]], color="black", lw=2)
        ax.plot([rotors[2, 0], rotors[3, 0]], [rotors[2, 1], rotors[3, 1]],
                [rotors[2, 2], rotors[3, 2]], color="black", lw=2)
        ax.scatter(rotors[:, 0], rotors[:, 1], rotors[:, 2], color="tab:blue", s=40)
        up_tip = pos + 0.35 * R[:, 2]
        ax.plot([pos[0], up_tip[0]], [pos[1], up_tip[1]], [pos[2], up_tip[2]],
                color="tab:orange", lw=2)

        ax.plot(tr[:, 0], tr[:, 1], tr[:, 2], color="tab:blue", alpha=0.4, lw=1)
        ax.scatter([tgt[0]], [tgt[1]], [tgt[2]], color="red", marker="*", s=200)

        # Keep the drone centered in a fixed-size cube so it "flies around".
        span = 3.0
        ax.set_xlim(pos[0] - span, pos[0] + span)
        ax.set_ylim(pos[1] - span, pos[1] + span)
        ax.set_zlim(max(0, pos[2] - span), pos[2] + span)
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_zlabel("z (m)")
        hits = int(env.waypoints_hit[0])
        title = f"{args.task}  |  dist {np.linalg.norm(tgt - pos):.2f} m"
        if args.task == "waypoint":
            title += f"  |  hits: {hits}"
            if env.manual_mode:
                title += f"  |  queued: {len(env.manual_queue)} [manual]"
        ax.set_title(title)

    if args.task == "waypoint":
        print("Manual waypoint controls (focus the window, then press):")
        print("  w/s = forward/back   a/d = left/right   r/f = up/down")
        print("  space = random waypoint   c = clear queue & hover here")
        print("Each key drops a waypoint; the drone flies your queue in order.\n")

    steps = int(args.seconds / dt) if args.seconds > 0 else None
    i = 0
    try:
        while plt.fignum_exists(fig.number):
            action, _, _ = net.act(obs, deterministic=True)
            obs, _, terminated, _, _ = env.step(action)
            if terminated[0]:
                trail.clear()
            if i % 2 == 0:  # render every other step (~25 fps) to stay smooth
                draw()
                plt.pause(max(1e-3, 2 * dt / args.speed))
            i += 1
            if steps is not None and i >= steps:
                break
    except KeyboardInterrupt:
        pass
    plt.ioff()
    print("done.")


if __name__ == "__main__":
    main()

"""Plotting: training curves and 3D flight visualizations (PNG + GIF)."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter


def plot_training(log: list[dict], path: str):
    steps = [e["steps"] for e in log]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, key, title in zip(
        axes,
        ["mean_return", "mean_ep_len", "mean_final_dist"],
        ["Episode return", "Episode length (steps)", "Distance to target (m)"],
    ):
        ax.plot(steps, [e[key] for e in log], lw=1.5)
        ax.set_title(title)
        ax.set_xlabel("environment steps")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_trajectory(positions: np.ndarray, targets: np.ndarray, path: str):
    """Static 3D plot. positions: [T,3], targets: [T,3] (may change over time)."""
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(projection="3d")
    ax.plot(*positions.T, lw=1.2, label="flight path")
    ax.scatter(*positions[0], color="green", s=40, label="start")
    uniq = np.unique(targets, axis=0)
    ax.scatter(uniq[:, 0], uniq[:, 1], uniq[:, 2], color="red", marker="*",
               s=120, label="target(s)")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_zlabel("z (m)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def animate_flight(positions: np.ndarray, targets: np.ndarray,
                   up_vectors: np.ndarray, path: str,
                   control_dt: float = 0.02, stride: int = 2):
    """Save an animated GIF of the flight. Arrays are [T,3]."""
    pos = positions[::stride]
    tgt = targets[::stride]
    up = up_vectors[::stride]

    fig = plt.figure(figsize=(6, 5.5))
    ax = fig.add_subplot(projection="3d")
    lo = np.minimum(pos.min(0), tgt.min(0)) - 0.5
    hi = np.maximum(pos.max(0), tgt.max(0)) + 0.5
    span = np.maximum(hi - lo, 2.0)
    center = (hi + lo) / 2
    lo, hi = center - span.max() / 2, center + span.max() / 2

    trail, = ax.plot([], [], [], lw=1.0, color="tab:blue", alpha=0.7)
    body, = ax.plot([], [], [], "o", color="black", ms=6)
    heading, = ax.plot([], [], [], lw=2.0, color="tab:orange")
    target_sc = ax.scatter([], [], [], color="red", marker="*", s=140)
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(max(0, lo[2]), hi[2])
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")

    def frame(i):
        trail.set_data(pos[:i + 1, 0], pos[:i + 1, 1])
        trail.set_3d_properties(pos[:i + 1, 2])
        body.set_data([pos[i, 0]], [pos[i, 1]])
        body.set_3d_properties([pos[i, 2]])
        tip = pos[i] + 0.4 * up[i]
        heading.set_data([pos[i, 0], tip[0]], [pos[i, 1], tip[1]])
        heading.set_3d_properties([pos[i, 2], tip[2]])
        target_sc._offsets3d = ([tgt[i, 0]], [tgt[i, 1]], [tgt[i, 2]])
        ax.set_title(f"t = {i * stride * control_dt:.1f} s")
        return trail, body, heading, target_sc

    anim = FuncAnimation(fig, frame, frames=len(pos), interval=1000 * control_dt * stride)
    anim.save(path, writer=PillowWriter(fps=int(1 / (control_dt * stride))))
    plt.close(fig)

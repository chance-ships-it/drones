"""Train a PPO policy to fly a quadrotor.

Examples:
  python train.py --task hover
  python train.py --task waypoint --updates 600 --num-envs 2048
  python train.py --device cpu          # skip MPS
"""

import argparse
import os

import torch

from drone_sim.env import QuadrotorEnv
from drone_sim.ppo import PPOConfig, PPOTrainer
from drone_sim.viz import plot_training


def pick_device(arg: str) -> str:
    if arg != "auto":
        return arg
    # MPS works but is ~5x slower than CPU for this workload: the physics
    # substeps are many small kernels, and GPU dispatch overhead dominates.
    # Benchmarked on M3: ~150k steps/s CPU vs ~30k steps/s MPS at 1024 envs.
    return "cpu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["hover", "waypoint"], default="hover")
    ap.add_argument("--num-envs", type=int, default=1024)
    ap.add_argument("--updates", type=int, default=300)
    ap.add_argument("--rollout-steps", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--device", default="auto", help="auto | mps | cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs")
    args = ap.parse_args()

    device = pick_device(args.device)
    torch.manual_seed(args.seed)
    print(f"task={args.task}  device={device}  envs={args.num_envs}  "
          f"updates={args.updates}")

    env = QuadrotorEnv(num_envs=args.num_envs, task=args.task, device=device,
                       seed=args.seed)
    cfg = PPOConfig(updates=args.updates, rollout_steps=args.rollout_steps,
                    lr=args.lr)
    trainer = PPOTrainer(env, cfg, device=device)

    run_dir = os.path.join(args.out, args.task)
    os.makedirs(run_dir, exist_ok=True)
    ckpt = os.path.join(run_dir, "policy.pt")

    def checkpoint(tr, entry):
        if entry["update"] % 25 == 0:
            tr.save(ckpt)

    trainer.train(log_every=10, on_update=checkpoint)
    trainer.save(ckpt)
    plot_training(trainer.log, os.path.join(run_dir, "training_curves.png"))
    print(f"saved policy -> {ckpt}")
    print(f"saved curves -> {run_dir}/training_curves.png")


if __name__ == "__main__":
    main()

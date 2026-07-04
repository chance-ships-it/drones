"""Vectorized quadrotor RL environment (Gymnasium-style API, batched tensors).

Tasks:
  hover     — reach and hold a fixed target point
  waypoint  — fly through a sequence of randomly placed waypoints
  recovery  — thrown in tumbling/inverted with high spin; must right itself
              and stabilize into a hover. Aggressive, nonlinear flight regime.

Observation (22 dims):
  target - pos (3, clipped), vel (3), body->world rotation matrix (9),
  omega (3), previous action (4)
Action (4 dims): normalized rotor thrust commands in [-1, 1].
"""

import torch

from drone_sim.physics import QuadrotorParams, QuadrotorPhysics, quat_to_rotmat

OBS_DIM = 22
ACT_DIM = 4


class QuadrotorEnv:
    def __init__(
        self,
        num_envs: int = 1024,
        task: str = "hover",
        device: str = "cpu",
        params: QuadrotorParams | None = None,
        episode_seconds: float = 8.0,
        arena_half_extent: float = 5.0,
        waypoint_radius: float = 0.35,
        seed: int | None = None,
    ):
        assert task in ("hover", "waypoint", "recovery")
        if seed is not None:
            torch.manual_seed(seed)
        self.n = num_envs
        self.task = task
        self.device = torch.device(device)
        self.sim = QuadrotorPhysics(num_envs, params, device)
        self.max_steps = int(episode_seconds / self.sim.p.control_dt)
        self.arena = arena_half_extent
        self.wp_radius = waypoint_radius

        # Recovery-task difficulty. `recovery_difficulty` in [0, 1] is a
        # curriculum knob: 0 = mild tumble, 1 = violently thrown, fully random
        # orientation. set_difficulty() ramps it during training.
        self.recovery_difficulty = 1.0
        self.recovery_max_tilt = 3.1       # radians of attitude spread at diff=1
        self.recovery_max_spin = 12.0      # rad/s of initial spin at diff=1
        self.upright_streak = torch.zeros(num_envs, device=self.device)

        n, dev = num_envs, self.device
        self.target = torch.zeros(n, 3, device=dev)
        self.step_count = torch.zeros(n, dtype=torch.long, device=dev)
        self.prev_action = torch.zeros(n, ACT_DIM, device=dev)
        self.waypoints_hit = torch.zeros(n, dtype=torch.long, device=dev)
        self.prev_dist = torch.zeros(n, device=dev)

        # Manual waypoint control (used by the live viewer). When manual_mode is
        # on, the env stops sampling random targets: it flies the queued points
        # in order and then holds position at the last one.
        self.manual_queue: list[torch.Tensor] = []
        self.manual_mode = False

    # ------------------------------------------------------------------ reset

    def queue_waypoint(self, xyz, env_id: int = 0):
        """Append a waypoint for the drone to fly to (enables manual mode).

        The drone visits queued waypoints in order, then holds at the last one.
        Used by the live viewer; also callable from your own scripts.
        """
        pt = torch.as_tensor(xyz, dtype=torch.float32, device=self.device)
        if not self.manual_mode:
            # First manual waypoint: redirect immediately instead of waiting for
            # the drone to reach whatever random target it currently has.
            self.manual_mode = True
            self.target[env_id] = pt
        else:
            self.manual_queue.append(pt)

    def set_difficulty(self, d: float):
        """Set recovery curriculum difficulty in [0, 1] (0=easy, 1=hardest)."""
        self.recovery_difficulty = float(max(0.0, min(1.0, d)))

    def _sample_targets(self, m: int) -> torch.Tensor:
        t = torch.empty(m, 3, device=self.device)
        t[:, :2] = (torch.rand(m, 2, device=self.device) * 2 - 1) * (self.arena * 0.5)
        t[:, 2] = 1.0 + torch.rand(m, device=self.device) * 2.0  # 1–3 m altitude
        return t

    def reset(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        if env_ids is None:
            env_ids = torch.arange(self.n, device=self.device)
        m = env_ids.shape[0]
        if m > 0:
            if self.task == "recovery":
                self._reset_recovery(env_ids, m)
            else:
                # Spawn near, not at, the target so the policy sees approach.
                start = self._sample_targets(m)
                offset = torch.randn(m, 3, device=self.device) * 0.8
                offset[:, 2] = offset[:, 2].clamp(min=-start[:, 2] + 0.3)
                vel0 = torch.randn(m, 3, device=self.device) * 0.3
                self.sim.reset(env_ids, start + offset, vel=vel0, tilt_std=0.15)
                self.target[env_ids] = start
            self.step_count[env_ids] = 0
            self.prev_action[env_ids] = 0.0
            self.waypoints_hit[env_ids] = 0
            self.upright_streak[env_ids] = 0.0
            self.prev_dist[env_ids] = (self.target[env_ids]
                                       - self.sim.pos[env_ids]).norm(dim=-1)
        return self._obs()

    def _reset_recovery(self, env_ids, m):
        """Throw the drone: high altitude, random orientation, high spin."""
        d = self.recovery_difficulty
        # Hover target at a fixed altitude directly below the throw point, so
        # the drone must both right itself AND arrest its fall/drift.
        start = torch.zeros(m, 3, device=self.device)
        start[:, :2] = (torch.rand(m, 2, device=self.device) * 2 - 1) * 1.5
        start[:, 2] = 3.0 + torch.rand(m, device=self.device) * 1.5  # 3–4.5 m up
        target = start.clone()
        target[:, 2] = 2.0                                  # recover to 2 m hover
        # Thrown with a tumble + kick that scale with difficulty.
        tilt = self.recovery_max_tilt * (0.2 + 0.8 * d)
        spin = self.recovery_max_spin * (0.2 + 0.8 * d)
        omega0 = (torch.rand(m, 3, device=self.device) * 2 - 1) * spin
        vel0 = torch.randn(m, 3, device=self.device) * (1.5 * (0.2 + 0.8 * d))
        self.sim.reset(env_ids, start, vel=vel0, tilt_std=tilt, omega=omega0)
        self.target[env_ids] = target

    # ------------------------------------------------------------------- step

    def step(self, action: torch.Tensor):
        action = action.clamp(-1.0, 1.0)
        self.sim.step(action)
        self.step_count += 1

        pos_err = self.target - self.sim.pos
        dist = pos_err.norm(dim=-1)
        progress = self.prev_dist - dist  # >0 when closing in on the target
        up_z = self.sim.up_vector[:, 2]

        # Waypoint task: advance target when close enough.
        reached = dist < self.wp_radius
        if self.task == "waypoint":
            ids = reached.nonzero(as_tuple=True)[0]
            if ids.numel() > 0:
                if self.manual_mode:
                    # Serve queued waypoints; hold position when the queue is
                    # empty (don't re-count the same held target every step).
                    for e in ids.tolist():
                        if self.manual_queue:
                            self.target[e] = self.manual_queue.pop(0)
                            self.waypoints_hit[e] += 1
                else:
                    self.target[ids] = self._sample_targets(ids.shape[0])
                    self.waypoints_hit[ids] += 1

        reward = self._reward(dist, up_z, action, progress, reached)
        # Measure against the (possibly resampled) target so the jump after a
        # capture doesn't register as a huge negative progress next step.
        self.prev_dist = (self.target - self.sim.pos).norm(dim=-1)

        if self.task == "recovery":
            # Inversion is the whole point here, so only the ground is fatal.
            # Track how long the drone has held an upright, calm attitude.
            stable = (up_z > 0.85) & (self.sim.omega.norm(dim=-1) < 1.5)
            self.upright_streak = torch.where(
                stable, self.upright_streak + 1, torch.zeros_like(self.upright_streak))
            crashed = self.sim.pos[:, 2] < 0.05
        else:
            crashed = (self.sim.pos[:, 2] < 0.05) | (up_z < 0.0)
        out_of_bounds = (dist > self.arena) | (self.sim.pos[:, 2] > 2 * self.arena)
        terminated = crashed | out_of_bounds
        truncated = self.step_count >= self.max_steps
        reward = reward - 25.0 * terminated.float()

        self.prev_action = action.clone()
        done_ids = (terminated | truncated).nonzero(as_tuple=True)[0]
        final_obs = self._obs()    # observation before autoreset, for bootstrapping
        info = {
            "dist": dist,
            "waypoints_hit": self.waypoints_hit.clone(),
            "upright_streak": self.upright_streak.clone(),
            "done_ids": done_ids,
            "final_obs": final_obs,
        }
        self.reset(done_ids)
        next_obs = self._obs() if done_ids.numel() > 0 else final_obs
        return next_obs, reward, terminated, truncated, info

    # ---------------------------------------------------------------- reward

    def _reward(self, dist, up_z, action, progress, reached):
        smooth_pen = 0.10 * (action - self.prev_action).norm(dim=-1)
        alive = 0.3

        if self.task == "recovery":
            spin = self.sim.omega.norm(dim=-1)
            speed = self.sim.vel.norm(dim=-1)
            # Priority ladder: (1) get upright, (2) stop spinning, (3) hold the
            # hover point. Weights reflect that order — righting dominates.
            upright_r = 1.5 * (0.5 * up_z + 0.5)             # 0 inverted .. 1.5 level
            spin_pen = 0.03 * spin
            pos_r = torch.exp(-0.5 * dist)                   # 1 at target, decays
            vel_pen = 0.03 * speed
            # Big bonus for actually reaching the recovered state (upright+calm),
            # so the policy is pulled toward *finishing* the recovery, not just
            # slowing the tumble.
            recovered = (up_z > 0.85) & (spin < 1.5)
            return (alive + upright_r + pos_r + 2.0 * recovered.float()
                    - spin_pen - vel_pen - smooth_pen)

        upright = 0.3 * up_z.clamp(min=0.0)
        spin_pen = 0.04 * self.sim.omega.norm(dim=-1)
        r = alive + upright - spin_pen - smooth_pen
        if self.task == "hover":
            pos_r = torch.exp(-1.2 * dist)                   # 1 at target, ~0 far
            vel_pen = 0.06 * self.sim.vel.norm(dim=-1)
            r = r + pos_r - vel_pen
        else:
            # Progress-based shaping: hovering short of the capture radius
            # earns nothing, so the only way to keep scoring is to keep
            # capturing waypoints. (Proximity shaping here gets exploited:
            # the policy parks just outside the radius and farms it.)
            speed = self.sim.vel.norm(dim=-1)
            # Soft speed cap: fast flight is fine, but reckless dashes that
            # flip the drone on the next hard turn are not. Only penalize
            # above 5 m/s so normal cruising is unaffected.
            overspeed = 0.5 * (speed - 5.0).clamp(min=0.0)
            r = r + 10.0 * progress + 15.0 * reached.float() - overspeed
        return r

    # ------------------------------------------------------------------- obs

    def _obs(self) -> torch.Tensor:
        rel = (self.target - self.sim.pos).clamp(-5.0, 5.0)
        rot = quat_to_rotmat(self.sim.quat).reshape(self.n, 9)
        return torch.cat(
            [rel, self.sim.vel, rot, self.sim.omega, self.prev_action], dim=-1
        )

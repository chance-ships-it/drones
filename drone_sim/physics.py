"""Batched quadrotor rigid-body dynamics in PyTorch.

All state is stored as [N, ...] tensors so thousands of drones simulate in
parallel on CPU or Apple-silicon GPU (MPS). Model: X-configuration quadrotor
with first-order motor lag, rotor thrust + yaw drag torque, linear body drag,
quaternion attitude, semi-implicit Euler integration with substeps.

Conventions:
  world frame: z up, gravity = -z
  quaternion q = [w, x, y, z], rotates body -> world
  angular velocity omega expressed in body frame
"""

from dataclasses import dataclass, field

import torch


@dataclass
class QuadrotorParams:
    mass: float = 0.75            # kg
    arm_length: float = 0.125     # m, center to rotor
    inertia: tuple = (2.3e-3, 2.3e-3, 4.0e-3)  # kg m^2, body-frame diagonal
    thrust_to_weight: float = 2.75  # total max thrust / weight
    yaw_torque_coeff: float = 0.016  # Nm of yaw torque per N of rotor thrust
    motor_tau: float = 0.033      # s, first-order motor response time
    linear_drag: float = 0.12     # N per (m/s), isotropic body drag
    angular_drag: float = 2.0e-4  # Nm per (rad/s)
    dt: float = 0.004             # s, physics substep
    substeps: int = 5             # substeps per control step (control at 50 Hz)
    gravity: float = 9.81

    # Per-episode domain randomization (fractional ranges, 0 disables).
    randomize: bool = True
    mass_range: float = 0.20
    thrust_range: float = 0.10
    drag_range: float = 0.30

    @property
    def control_dt(self) -> float:
        return self.dt * self.substeps


def quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vectors v [N,3] by quaternions q [N,4] (body -> world)."""
    w, xyz = q[:, :1], q[:, 1:]
    t = 2.0 * torch.cross(xyz, v, dim=-1)
    return v + w * t + torch.cross(xyz, t, dim=-1)


def quat_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    """Quaternions [N,4] -> rotation matrices [N,3,3] (body -> world)."""
    w, x, y, z = q.unbind(-1)
    xx, yy, zz = x * x, y * y, z * z
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z
    m = torch.stack(
        [
            1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy),
            2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx),
            2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy),
        ],
        dim=-1,
    )
    return m.view(-1, 3, 3)


def quat_integrate(q: torch.Tensor, omega: torch.Tensor, dt: float) -> torch.Tensor:
    """Integrate q_dot = 0.5 * q ⊗ [0, omega_body] and renormalize."""
    w, x, y, z = q.unbind(-1)
    ox, oy, oz = omega.unbind(-1)
    dq = 0.5 * torch.stack(
        [
            -x * ox - y * oy - z * oz,
            w * ox + y * oz - z * oy,
            w * oy - x * oz + z * ox,
            w * oz + x * oy - y * ox,
        ],
        dim=-1,
    )
    q = q + dq * dt
    return q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)


class QuadrotorPhysics:
    """Vectorized simulator for N quadrotors."""

    def __init__(self, num_envs: int, params: QuadrotorParams | None = None,
                 device: str = "cpu"):
        self.n = num_envs
        self.p = params or QuadrotorParams()
        self.device = torch.device(device)

        n, dev = num_envs, self.device
        self.pos = torch.zeros(n, 3, device=dev)
        self.quat = torch.zeros(n, 4, device=dev)
        self.quat[:, 0] = 1.0
        self.vel = torch.zeros(n, 3, device=dev)
        self.omega = torch.zeros(n, 3, device=dev)
        # Current rotor thrusts in N, order: [front, left, back, right] (X config)
        self.rotor_thrust = torch.zeros(n, 4, device=dev)

        # Per-env randomized physical properties (filled by randomize()).
        self.mass = torch.full((n, 1), self.p.mass, device=dev)
        self.inertia = torch.tensor(self.p.inertia, device=dev).expand(n, 3).clone()
        self.thrust_gain = torch.ones(n, 1, device=dev)
        self.drag_gain = torch.ones(n, 1, device=dev)

        hover_total = self.p.mass * self.p.gravity
        self.max_thrust_per_rotor = self.p.thrust_to_weight * hover_total / 4.0

        # Rotor geometry: X config, arms at 45 deg. Signs give roll/pitch/yaw mixing.
        L = self.p.arm_length / (2.0 ** 0.5)
        self.rotor_x = torch.tensor([L, -L, -L, L], device=dev)      # body x offsets
        self.rotor_y = torch.tensor([-L, -L, L, L], device=dev)     # body y offsets
        self.rotor_spin = torch.tensor([1.0, -1.0, 1.0, -1.0], device=dev)

    def randomize(self, env_ids: torch.Tensor):
        """Resample physical properties for the given envs (domain randomization)."""
        if not self.p.randomize:
            return
        m = env_ids.shape[0]
        if m == 0:
            return

        def uniform(scale):
            return 1.0 + scale * (2.0 * torch.rand(m, 1, device=self.device) - 1.0)

        self.mass[env_ids] = self.p.mass * uniform(self.p.mass_range)
        self.thrust_gain[env_ids] = uniform(self.p.thrust_range)
        self.drag_gain[env_ids] = uniform(self.p.drag_range)

    def reset(self, env_ids: torch.Tensor, pos: torch.Tensor,
              vel: torch.Tensor | None = None, tilt_std: float = 0.0):
        """Reset selected envs to given positions with small random attitude."""
        m = env_ids.shape[0]
        self.pos[env_ids] = pos
        self.vel[env_ids] = vel if vel is not None else torch.zeros(m, 3, device=self.device)
        self.omega[env_ids] = 0.0

        q = torch.zeros(m, 4, device=self.device)
        q[:, 0] = 1.0
        if tilt_std > 0.0:
            axis_angle = tilt_std * torch.randn(m, 3, device=self.device)
            angle = axis_angle.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            axis = axis_angle / angle
            q = torch.cat([torch.cos(angle / 2), axis * torch.sin(angle / 2)], dim=-1)
        self.quat[env_ids] = q

        # Start motors at per-env hover thrust so episodes don't begin in freefall.
        hover = (self.mass[env_ids] * self.p.gravity / 4.0) / self.thrust_gain[env_ids]
        self.rotor_thrust[env_ids] = hover.expand(m, 4).clamp(0.0, self.max_thrust_per_rotor)
        self.randomize(env_ids)

    def step(self, action: torch.Tensor):
        """Advance one control step. action: [N,4] normalized thrusts in [-1, 1]."""
        cmd = (action.clamp(-1.0, 1.0) + 1.0) * 0.5 * self.max_thrust_per_rotor
        alpha = 1.0 - torch.exp(torch.tensor(-self.p.dt / self.p.motor_tau))

        for _ in range(self.p.substeps):
            # First-order motor lag toward commanded thrust.
            self.rotor_thrust = self.rotor_thrust + alpha * (cmd - self.rotor_thrust)
            thrust = self.rotor_thrust * self.thrust_gain  # [N,4] effective N

            # Body-frame torques from rotor placement and spin drag.
            tau_x = (thrust * self.rotor_y).sum(-1, keepdim=True) * -1.0
            tau_y = (thrust * self.rotor_x).sum(-1, keepdim=True)
            tau_z = (thrust * self.rotor_spin).sum(-1, keepdim=True) * self.p.yaw_torque_coeff
            tau = torch.cat([tau_x, tau_y, tau_z], dim=-1)
            tau = tau - self.p.angular_drag * self.omega

            # World-frame forces: rotor thrust along body z, gravity, linear drag.
            body_z = quat_rotate(self.quat, torch.tensor([[0.0, 0.0, 1.0]], device=self.device).expand(self.n, 3))
            force = thrust.sum(-1, keepdim=True) * body_z
            force = force - self.p.linear_drag * self.drag_gain * self.vel
            force[:, 2] -= (self.mass * self.p.gravity).squeeze(-1)

            # Euler's rotation equation with diagonal inertia (body frame).
            Iw = self.inertia * self.omega
            omega_dot = (tau - torch.cross(self.omega, Iw, dim=-1)) / self.inertia

            # Semi-implicit Euler.
            self.vel = self.vel + (force / self.mass) * self.p.dt
            self.omega = self.omega + omega_dot * self.p.dt
            self.pos = self.pos + self.vel * self.p.dt
            self.quat = quat_integrate(self.quat, self.omega, self.p.dt)

    @property
    def up_vector(self) -> torch.Tensor:
        """World-frame body-z axis [N,3]; z component is cos(tilt)."""
        return quat_to_rotmat(self.quat)[:, :, 2]

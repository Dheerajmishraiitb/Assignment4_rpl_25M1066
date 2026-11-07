#!/usr/bin/env python3
"""
Flow-matching math - Part 3 of the assignment. Pure functions, no ROS, so
they're unit-testable without the simulator (see the bottom of this file
for a self-check: `python3 fm_losses.py`).

Conditional flow matching with the straight-line (rectified-flow) path:

    a_0 ~ N(0, I)                 noise
    a_1                           a demonstrated action (normalised)
    a_t = (1 - t) a_0 + t a_1     a point on the straight path, t in [0, 1]
    u_t = a_1 - a_0               its velocity (constant along the path)

The network v(a_t, t | obs) is trained to regress u_t. At test time an
action is drawn by starting from fresh noise and integrating da/dt = v
from t = 0 to t = 1. Different noise lands on different modes of the
demonstrated actions, instead of on their average the way an MSE
regression head does.
"""

import torch
import torch.nn.functional as F


class FlowMatchingSchedule:
    """The interpolation path between noise and data, and its velocity."""

    def sample_t(self, n, device=None):
        """n training times, uniform on [0, 1]."""
        return torch.rand(n, device=device)

    def interpolate(self, a0, a1, t):
        """a0, a1: (B, *D). t: (B,). -> a_t = (1 - t) a0 + t a1, per sample
        (reshape t so it broadcasts over every action dimension)."""
        # reshape t (B,) -> (B, 1, ..., 1) so it broadcasts over action dims
        t = t.reshape(-1, *([1] * (a0.dim() - 1)))
        return (1.0 - t) * a0 + t * a1

    def target_velocity(self, a0, a1):
        """d a_t / dt along the straight path - independent of t."""
        return a1 - a0


def flow_matching_loss(model, schedule, img, odom, a1):
    """One flow-matching loss value for a batch of demonstrated actions a1
    (B, *D), observed with img / odom. model(img, odom, a_t, t) returns a
    predicted velocity, same shape as a1.

    Draw noise a0 ~ N(0, I) like a1, times t with schedule.sample_t, build
    a_t with schedule.interpolate, and return the mean squared error
    between the model's velocity at (a_t, t) and
    schedule.target_velocity(a0, a1)."""
    a0 = torch.randn_like(a1)                              # noise
    t = schedule.sample_t(a1.shape[0], device=a1.device)   # (B,)
    a_t = schedule.interpolate(a0, a1, t)
    pred = model(img, odom, a_t, t)
    return F.mse_loss(pred, schedule.target_velocity(a0, a1))


@torch.no_grad()
def sample_actions(model, img, odom, act_shape, steps=10, noise=None):
    """Euler-integrate da/dt = v(a, t | obs) from noise (t=0) to t=1.
    act_shape: per-sample action shape, e.g. (2,). noise: optional (B,
    *act_shape) starting point; fresh N(0, I) if None (do not modify the
    caller's tensor). B = odom.shape[0]; put new tensors on odom.device.

    `steps` equal steps of dt = 1/steps: at step k (k = 0 .. steps-1) the
    time is t = k*dt (a (B,) tensor) and a <- a + dt * model(img, odom, a, t).
    -> the final a, (B, *act_shape)."""
    B = odom.shape[0]
    if noise is None:
        a = torch.randn((B, *act_shape), device=odom.device)
    else:
        a = noise.clone().to(odom.device)        # never modify the caller's tensor
    dt = 1.0 / steps
    for k in range(steps):
        t = torch.full((B,), k * dt, device=odom.device)
        a = a + dt * model(img, odom, a, t)
    return a


if __name__ == '__main__':
    # Offline self-check - no ROS/Gazebo needed.
    s = FlowMatchingSchedule()
    a0 = torch.tensor([[0.0, 0.0], [1.0, -1.0]])
    a1 = torch.tensor([[2.0, 4.0], [3.0, 1.0]])
    t = torch.tensor([0.25, 1.0])
    at = s.interpolate(a0, a1, t)
    assert torch.allclose(at, torch.tensor([[0.5, 1.0], [3.0, 1.0]])), at
    assert torch.allclose(s.target_velocity(a0, a1), a1 - a0)
    print('interpolate / target_velocity: OK')

    # A model that outputs the exact velocity toward a fixed target must
    # carry any noise exactly onto that target.
    target = torch.tensor([0.3, -0.7])

    def exact(img, odom, a, t):
        return (target - a) / (1.0 - t).clamp(min=1e-6).unsqueeze(-1)

    out = sample_actions(exact, None, torch.zeros(5, 4), (2,), steps=20)
    assert torch.allclose(out, target.expand(5, 2), atol=1e-4), out
    print('sample_actions: OK')

    # Bimodal 1-D toy: data is +-1 with no conditioning. A tiny flow must
    # put its samples near +-1, NOT near the mean 0 that MSE would give.
    torch.manual_seed(0)
    net = torch.nn.Sequential(torch.nn.Linear(2, 64), torch.nn.SiLU(),
                              torch.nn.Linear(64, 64), torch.nn.SiLU(),
                              torch.nn.Linear(64, 1))

    def toy(img, odom, a, t):
        return net(torch.cat([a, t.unsqueeze(-1)], dim=-1))

    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    for _ in range(1500):
        data = (torch.randint(0, 2, (256, 1)) * 2 - 1).float()
        loss = flow_matching_loss(toy, s, None, torch.zeros(256, 4), data)
        opt.zero_grad()
        loss.backward()
        opt.step()
    smp = sample_actions(toy, None, torch.zeros(2000, 4), (1,), steps=20)
    near_mode = ((smp.abs() - 1).abs() < 0.3).float().mean().item()
    near_mean = (smp.abs() < 0.3).float().mean().item()
    assert near_mode > 0.8 and near_mean < 0.1, (near_mode, near_mean)
    print(f'bimodal toy: OK ({near_mode:.0%} of samples near +-1, '
          f'{near_mean:.0%} near the mean 0)')
    print('\nall checks passed')

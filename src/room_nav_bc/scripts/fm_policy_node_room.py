#!/usr/bin/env python3
"""
Run a trained flow-matching policy (fm_train_room.py) in closed loop and
score it - the Part 3 counterpart of bc_policy_node_room.py, and graded by
exactly the same node (success = stopped inside the goal, collision check
against every obstacle and wall), so results compare directly with Part 1.

    ros2 run room_nav_bc fm_policy_node_room.py --policy policy_room_fm.pt \
        --starts test --trials 3

Each control step draws an action by integrating the learned velocity
field from noise (--steps Euler steps). Where the noise comes from:

    --noise step    fresh noise every control step (default). The policy
                    may sample "left" on one step and "right" on the next;
                    closed-loop feedback usually settles it onto one side.
    --noise trial   one noise draw per trial, reused every step - the same
                    policy, but it tends to commit to one mode for the
                    whole run.
    --noise zero    always start from a_0 = 0, the centre of the noise
                    distribution - a deterministic policy, roughly "the
                    most typical" action. Worth comparing: it can land
                    between the modes, like regression.
"""

import os
import sys

import numpy as np
import torch
import yaml

import rclpy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bc_policy_node_room import (  # noqa: E402
    PolicyRunner, build_parser as bc_build_parser, default_config_path,
    load_poses,
)
from fm_train_room import FlowPolicy  # noqa: E402


class FlowSampler:
    """Callable like a regression CNNPolicy - (img, odom) -> normalised
    action - so bc_policy_node_room.py's PolicyRunner drives it unchanged."""

    def __init__(self, model, steps, noise, seed):
        self.model = model
        self.steps = steps
        self.noise = noise
        self.gen = torch.Generator().manual_seed(seed)
        self.trial_noise = None
        self.new_trial()

    def new_trial(self):
        self.trial_noise = torch.randn(1, *self.model.act_shape(),
                                       generator=self.gen)

    def __call__(self, img, odom):
        if self.noise == 'trial':
            a0 = self.trial_noise
        elif self.noise == 'zero':
            a0 = torch.zeros(1, *self.model.act_shape())
        else:
            a0 = torch.randn(1, *self.model.act_shape(), generator=self.gen)
        return self.model.sample(img, odom, self.steps, a0)


class FlowRunner(PolicyRunner):

    def begin_trial(self):
        self.model.new_trial()
        super().begin_trial()


def load_policy(path, steps, noise, seed):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    if ck.get('kind') != 'flow_matching':
        raise SystemExit(f"{path} is not a flow-matching checkpoint - use "
                         "bc_policy_node_room.py for regression policies.")
    model = FlowPolicy(ck['odom_dim'], ck['img_size'], ck['chunk'],
                       ck['img_embed'], tuple(ck['hidden']), ck['t_embed'])
    model.load_state_dict(ck['state_dict'])
    model.eval()
    steps = steps or int(ck.get('sample_steps', 10))
    return FlowSampler(model, steps, noise, seed), ck


def build_parser():
    p = bc_build_parser()
    p.description = __doc__
    p.set_defaults(policy='policy_room_fm.pt')
    p.add_argument('--steps', type=int, default=0,
                   help='Euler integration steps (0 = value saved at '
                        'training time)')
    p.add_argument('--noise', choices=['step', 'trial', 'zero'],
                   default='step')
    p.add_argument('--noise-seed', type=int, default=0)
    return p


def main():
    args = build_parser().parse_args()
    if args.config is None:
        args.config = default_config_path()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    poses = load_poses(args.config, args.starts)
    room = cfg['room']
    args.room_x_min, args.room_x_max = room['x_min'], room['x_max']
    args.room_y_min, args.room_y_max = room['y_min'], room['y_max']

    sampler, ck = load_policy(args.policy, args.steps, args.noise,
                              args.noise_seed)
    print(f"  flow matching: {sampler.steps} Euler steps, noise '{args.noise}'")

    rclpy.init()
    node = FlowRunner(args, sampler, ck, poses, cfg['obstacles'], cfg['goal'])
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, RuntimeError):
        pass
    finally:
        try:
            node.publish(0.0, 0.0)
            node.save_steps()
        except Exception:
            pass
        try:
            node.destroy_node()
        except Exception:
            pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

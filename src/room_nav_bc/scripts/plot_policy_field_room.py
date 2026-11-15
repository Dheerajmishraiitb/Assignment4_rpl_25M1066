#!/usr/bin/env python3
"""
Top-down map of what trained policies would do all over the room - Part 3
of the assignment. BC and flow matching side by side.

    # simulator running (gui:=false is fine) the FIRST time only
    ros2 run room_nav_bc plot_policy_field_room.py \
        --bc policy_room_human.pt --fm policy_room_fm.pt \
        --data-dir data_room_human --out figures/policy_field.png

The robot is teleported to every point of a grid over the room (--spacing,
skipping points inside obstacles), always facing --heading, and the camera
frame is saved at each one to --frames. If that file already exists the
simulator is not needed: delete it (or pass --recapture) after changing
the grid or heading.

Each policy then gets, at every grid point, the observation it would have
there: that camera frame plus odometry measured from the start pose
--start, exactly as if a trial had begun at --start and the robot had
driven to this point (dx, dy from the start; sin/cos of the heading).

Each arrow goes from the grid point to where the robot would be after
holding the commanded (v, w) for --horizon seconds. BC gives one action
per point; flow matching gives --samples, one per noise draw. A green dot
is an action that passes the stop test (|v| < 0.01 m/s, |w| < 0.05 rad/s).
Grey lines are the demonstrations the policies were trained on.

--rollouts BC.csv FM.csv overlays real closed-loop trials (the --steps-csv
of bc_policy_node_room.py / fm_policy_node_room.py), e.g. from an
out-of-distribution start:

    ros2 run room_nav_bc bc_policy_node_room.py --policy policy_room_human.pt \
        --starts ood --trials 3 --jitter-xy 0.02 --steps-csv logs/bc_ood.csv
    (same with fm_policy_node_room.py -> logs/fm_ood.csv)
    ros2 run room_nav_bc plot_policy_field_room.py --start 0.0,1.25 \
        --rollouts logs/bc_ood.csv logs/fm_ood.csv --out figures/policy_field_ood.png
"""

import argparse
import csv
import glob
import math
import os
import pickle
import sys
import time

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bc_dataset_room import decode_image  # noqa: E402
from bc_policy_node_room import (  # noqa: E402
    MAX_ANG, MAX_LIN, default_config_path, load_policy as load_bc,
    load_poses, quaternion_from_yaw, worst_clearance, yaw_from_quaternion,
)
from bc_train_room import to_img_tensor  # noqa: E402
from fm_policy_node_room import load_policy as load_fm  # noqa: E402
from record_demo_room import encode_frame  # noqa: E402

STOP_V, STOP_W = 0.01, 0.05      # the grader's stop test


# -- 1. capture: one camera frame per grid pose (needs Gazebo) ---------------

def grid_points(cfg, spacing, margin=0.05):
    """Grid over the room, keeping points the robot fits at."""
    room, obstacles = cfg['room'], cfg['obstacles']
    xs = np.arange(room['x_min'] + spacing / 2, room['x_max'], spacing)
    ys = np.arange(room['y_min'] + spacing / 2, room['y_max'], spacing)
    return [(float(x), float(y)) for y in ys for x in xs
            if worst_clearance(x, y, obstacles, room)[0] > margin]


def capture(args, points, yaw):
    import rclpy
    from cv_bridge import CvBridge
    from gazebo_msgs.msg import ModelStates
    from gazebo_msgs.srv import SetEntityState
    from geometry_msgs.msg import Twist
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import Image

    class Grabber(Node):
        def __init__(self):
            super().__init__('policy_field_capture')
            self.bridge = CvBridge()
            self.pose = self.frame = None
            self.frame_t = 0.0
            qos = QoSProfile(depth=10)
            qos.reliability = ReliabilityPolicy.BEST_EFFORT
            self.create_subscription(ModelStates, '/gazebo/model_states',
                                     self.on_states, qos)
            self.create_subscription(Image, args.image_topic, self.on_image,
                                     qos)
            self.pub = self.create_publisher(Twist, '/cmd_vel', 10)
            self.cli = self.create_client(SetEntityState,
                                          '/gazebo/set_entity_state')

        def on_states(self, msg):
            if args.robot in msg.name:
                p = msg.pose[msg.name.index(args.robot)]
                q = p.orientation
                self.pose = (p.position.x, p.position.y,
                             yaw_from_quaternion(q.x, q.y, q.z, q.w))

        def on_image(self, msg):
            self.frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            self.frame_t = time.time()

        def teleport(self, x, y, yaw):
            req = SetEntityState.Request()
            req.state.name = args.robot
            req.state.pose.position.x = x
            req.state.pose.position.y = y
            req.state.pose.position.z = 0.01
            (req.state.pose.orientation.x, req.state.pose.orientation.y,
             req.state.pose.orientation.z,
             req.state.pose.orientation.w) = quaternion_from_yaw(yaw)
            self.cli.call_async(req)

    rclpy.init()
    node = Grabber()
    if not node.cli.wait_for_service(timeout_sec=10.0):
        raise SystemExit("no /gazebo/set_entity_state - is the simulator "
                         "running? (ros2 launch room_nav_bc "
                         "room_demo.launch.py gui:=false)")
    jpegs, poses = [], []
    try:
        for k, (x, y) in enumerate(points):
            node.pub.publish(Twist())
            node.teleport(x, y, yaw)
            # Wait until the robot is really there, then take the first
            # frame rendered at least --settle s later (not a stale one).
            t0, t_there = time.time(), None
            while time.time() - t0 < 5.0:
                rclpy.spin_once(node, timeout_sec=0.02)
                p = node.pose
                if t_there is None and p is not None \
                        and math.hypot(p[0] - x, p[1] - y) < 0.02:
                    t_there = time.time()
                if t_there is not None \
                        and node.frame_t > t_there + args.settle:
                    break
            else:
                print(f"  ({x:+.2f}, {y:+.2f}): no fresh frame, skipped")
                continue
            jpegs.append(encode_frame(node.frame))
            poses.append(node.pose)
            print(f"\r  captured {k + 1}/{len(points)}", end='', flush=True)
    finally:
        node.pub.publish(Twist())
        node.destroy_node()
        rclpy.shutdown()
    print()
    return np.array(jpegs, dtype=object), np.asarray(poses, np.float32)


# -- 2. what each policy does there (offline) ---------------------------------

def observations(ck, jpegs, poses, start):
    """Camera frames + odometry from `start`, normalised like training.
    (In this simulator odometry = ground-truth pose, so dx, dy are just
    the position minus the start position - see the demo files.)"""
    img = to_img_tensor(np.stack([decode_image(j, ck['img_size'])
                                  for j in jpegs]), 'cpu')
    odom = np.stack([poses[:, 0] - start[0], poses[:, 1] - start[1],
                     np.sin(poses[:, 2]), np.cos(poses[:, 2])], 1)
    odom = (odom - np.asarray(ck['odom_mean'])) / np.asarray(ck['odom_std'])
    return img, torch.as_tensor(odom, dtype=torch.float32)


def to_command(a_n, ck):
    """Normalised action(s) -> (v, w) the node would publish (first step
    of a chunk, clipped to the robot's limits)."""
    a = a_n.numpy()
    if a.ndim == 3:
        a = a[:, 0]
    a = a * np.asarray(ck['act_std']) + np.asarray(ck['act_mean'])
    return np.stack([np.clip(a[:, 0], -MAX_LIN, MAX_LIN),
                     np.clip(a[:, 1], -MAX_ANG, MAX_ANG)], 1)


@torch.no_grad()
def bc_commands(path, jpegs, poses, start):
    """-> (N, 1, 2): BC's single action at every grid point."""
    model, ck = load_bc(path)
    img, odom = observations(ck, jpegs, poses, start)
    return to_command(model(img, odom), ck)[:, None]


@torch.no_grad()
def fm_commands(path, jpegs, poses, start, k, seed):
    """-> (N, k, 2): k flow-matching samples at every grid point."""
    sampler, ck = load_fm(path, 0, 'step', seed)
    torch.manual_seed(seed)
    img, odom = observations(ck, jpegs, poses, start)
    a = sampler.model.sample(img.repeat_interleave(k, 0),
                             odom.repeat_interleave(k, 0), sampler.steps)
    return to_command(a, ck).reshape(len(jpegs), k, 2)


def drive(pose, v, w, horizon):
    """Where holding (v, w) for `horizon` s takes the robot (exact arc)."""
    x, y, yaw = pose
    if abs(w) < 1e-6:
        return x + v * horizon * math.cos(yaw), y + v * horizon * math.sin(yaw)
    r, yaw1 = v / w, yaw + w * horizon
    return (x + r * (math.sin(yaw1) - math.sin(yaw)),
            y - r * (math.cos(yaw1) - math.cos(yaw)))


# -- 3. plot ------------------------------------------------------------------

def draw_room(ax, cfg):
    import matplotlib.patches as patches
    room, goal = cfg['room'], cfg['goal']
    ax.add_patch(patches.Rectangle(
        (room['x_min'], room['y_min']), room['x_max'] - room['x_min'],
        room['y_max'] - room['y_min'], fill=False, lw=2, color='black'))
    for b in cfg['obstacles'].values():
        ax.add_patch(patches.Rectangle(
            (b['x'] - b['w'] / 2, b['y'] - b['h'] / 2), b['w'], b['h'],
            color='#8b5a2b', alpha=0.6))
    ax.add_patch(patches.Rectangle(
        (goal['x'] - goal['w'] / 2, goal['y'] - goal['h'] / 2),
        goal['w'], goal['h'], color='gold', alpha=0.5, label='goal'))
    ax.set_xlim(room['x_min'] - 0.05, room['x_max'] + 0.05)
    ax.set_ylim(room['y_min'] - 0.05, room['y_max'] + 0.05)
    ax.set_aspect('equal')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')


def demo_paths(dirs):
    paths = []
    for d in dirs:
        for f in sorted(glob.glob(os.path.join(d, '*.p'))):
            with open(f, 'rb') as fh:
                s = pickle.load(fh)['samples']
            paths.append((np.array([r['x'] for r in s]),
                          np.array([r['y'] for r in s])))
    return paths


def rollouts(path):
    """steps CSV -> [(xs, ys, collided)] per trial, cut at the first
    contact (a trial keeps running after a collision, until its timeout)."""
    trials = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            xs, ys, hit = trials.setdefault(r['trial'], ([], [], [False]))
            if hit[0]:
                continue
            xs.append(float(r['x']))
            ys.append(float(r['y']))
            hit[0] = float(r['clearance']) < 0.0
    return [(np.array(xs), np.array(ys), hit[0])
            for xs, ys, hit in trials.values()]


def panel(ax, cfg, title, poses, cmds, demos, start, horizon, colour,
          runs=()):
    draw_room(ax, cfg)
    for i, (xs, ys) in enumerate(demos):
        ax.plot(xs, ys, color='0.6', lw=1, zorder=1,
                label='demonstrations' if i == 0 else None)
    k = cmds.shape[1]
    alpha = 1.0 if k == 1 else 0.45
    arrows, stops = [], []
    for pose, acts in zip(poses, cmds):
        for v, w in acts:
            if abs(v) < STOP_V and abs(w) < STOP_W:
                stops.append(pose[:2])
            else:
                ex, ey = drive(pose, v, w, horizon)
                arrows.append((pose[0], pose[1], ex - pose[0], ey - pose[1]))
    if arrows:
        a = np.array(arrows)
        ax.quiver(a[:, 0], a[:, 1], a[:, 2], a[:, 3], angles='xy',
                  scale_units='xy', scale=1, color=colour, alpha=alpha,
                  width=0.004, zorder=3)
    if stops:
        s = np.array(stops)
        ax.scatter(s[:, 0], s[:, 1], s=30, c='tab:green', zorder=4,
                   edgecolors='k', linewidths=0.5,
                   label='passes the stop test')
    for i, (xs, ys, hit) in enumerate(runs):
        ax.plot(xs, ys, color='tab:orange', lw=2.5, zorder=6,
                label='closed-loop trials' if i == 0 else None)
        if hit:
            ax.plot(xs[-1], ys[-1], marker='X', ms=11, c='tab:orange',
                    mec='k', zorder=7,
                    label='collided' if not any(r[2] for r in runs[:i])
                    else None)
    ax.plot(*start, marker='*', ms=14, c='tab:red', mec='k', zorder=5,
            label='start (odometry origin)')
    ax.set_title(f"{title}\n{k} action{'s' if k > 1 else ''} per point",
                 fontsize=10)


def stop_rate(cmds, poses, goal):
    """Fraction of actions passing the stop test at grid points in the goal."""
    inside = [i for i, (x, y, _) in enumerate(poses)
              if abs(x - goal['x']) <= goal['w'] / 2
              and abs(y - goal['y']) <= goal['h'] / 2]
    if not inside:
        return float('nan'), 0
    c = cmds[inside]
    ok = (np.abs(c[..., 0]) < STOP_V) & (np.abs(c[..., 1]) < STOP_W)
    return float(ok.mean()), len(inside)


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--bc', default='policy_room_human.pt')
    p.add_argument('--fm', default='policy_room_fm.pt')
    p.add_argument('--data-dir', nargs='*', default=['data_room_human'],
                   help='demonstrations to draw (any number of folders)')
    p.add_argument('--config', default=None)
    p.add_argument('--start', default='A',
                   help="odometry origin: a 'train' start label, or X,Y "
                        "in metres for any other start (e.g. 0.0,1.25, "
                        "right in front of the island)")
    p.add_argument('--heading', type=float, default=90.0,
                   help='deg, robot heading at every grid point '
                        '(90 = facing the goal)')
    p.add_argument('--spacing', type=float, default=0.2, help='m')
    p.add_argument('--samples', type=int, default=8,
                   help='flow-matching samples per point')
    p.add_argument('--horizon', type=float, default=0.8,
                   help='s, how far ahead each arrow looks')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--frames', default='figures/field_frames.npz',
                   help='camera frames per grid pose (captured if missing)')
    p.add_argument('--recapture', action='store_true')
    p.add_argument('--rollouts', nargs=2, metavar=('BC_CSV', 'FM_CSV'),
                   help='overlay closed-loop trials (--steps-csv files)')
    p.add_argument('--settle', type=float, default=0.3,
                   help='s to wait after teleporting before taking a frame')
    p.add_argument('--robot', default='burger')
    p.add_argument('--image-topic', default='/camera/image_raw')
    p.add_argument('--out', default='figures/policy_field.png')
    args = p.parse_args()

    args.config = args.config or default_config_path()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    starts = {lab: (x, y) for lab, x, y, _ in load_poses(args.config, 'train')}
    if args.start in starts:
        start = starts[args.start]
    else:
        try:
            start = tuple(float(c) for c in args.start.split(','))
            assert len(start) == 2
        except (ValueError, AssertionError):
            raise SystemExit(f"--start must be one of {', '.join(starts)} "
                             "or X,Y")

    if args.recapture or not os.path.exists(args.frames):
        points = grid_points(cfg, args.spacing)
        print(f"capturing {len(points)} grid poses, heading "
              f"{args.heading:.0f} deg (needs the simulator) ...")
        jpegs, poses = capture(args, points, math.radians(args.heading))
        os.makedirs(os.path.dirname(args.frames) or '.', exist_ok=True)
        np.savez(args.frames, jpegs=jpegs, poses=poses)
        print(f"saved {args.frames}")
    d = np.load(args.frames, allow_pickle=True)
    jpegs, poses = d['jpegs'], d['poses']

    bc = bc_commands(args.bc, jpegs, poses, start)
    fm = fm_commands(args.fm, jpegs, poses, start, args.samples, args.seed)
    for name, c in (('BC', bc), ('FM', fm)):
        rate, n = stop_rate(c, poses, cfg['goal'])
        print(f"{name}: {rate:.0%} of actions at the {n} grid points inside "
              f"the goal pass the stop test")

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    demos = demo_paths(args.data_dir)
    runs = [rollouts(f) for f in args.rollouts] if args.rollouts else [(), ()]
    fig, axes = plt.subplots(1, 2, figsize=(11, 7.5))
    panel(axes[0], cfg, f"BC ({os.path.basename(args.bc)})", poses, bc,
          demos, start, args.horizon, 'black', runs[0])
    panel(axes[1], cfg, f"Flow matching ({os.path.basename(args.fm)})",
          poses, fm, demos, start, args.horizon, 'tab:blue', runs[1])
    fig.suptitle(f"Where each policy would drive in the next "
                 f"{args.horizon:g} s - robot facing {args.heading:.0f} deg, "
                 f"odometry measured from start {args.start}")
    seen = {}
    for ax in axes:
        for h, lab in zip(*ax.get_legend_handles_labels()):
            seen.setdefault(lab, h)
    handles, labels = list(seen.values()), list(seen)
    fig.legend(handles, labels, loc='lower center', ncol=len(labels),
               fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print(f"saved {args.out}")


if __name__ == '__main__':
    main()

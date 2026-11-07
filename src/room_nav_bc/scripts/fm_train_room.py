#!/usr/bin/env python3
"""
Train a flow-matching behavioural-cloning policy on camera + odometry
room-navigation demonstrations - Part 3 of the assignment. Same dataset
as Part 1 (bc_dataset_room.py's output), same CNN encoder; only the head
changes, from regressing the action to predicting a flow-matching velocity
(see fm_losses.py for the math).

    python3 bc_dataset_room.py --data-dir data_room_human --out dataset_room_human.npz
    python3 fm_train_room.py   --dataset dataset_room_human.npz --out policy_room_fm.pt

WHY.  An L1/MSE head fits ONE action per observation. Where the
demonstrations split (left vs right of the island) and nothing in the
observation says which way an episode went, the best single answer is
somewhere between the two - straight into the island. A flow-matching head
models the whole action distribution instead: sampling starts from noise
and is carried onto one of the demonstrated modes, so it can commit to
"left" or "right" rather than average them.

A NOTE ON THE LOSS.  The flow-matching loss is the error in predicting a
NOISY velocity target (it depends on the random noise and t drawn for each
sample), so it never goes near zero and is not comparable to Part 1's
regression loss. Its validation value is also a poor stopping signal here
(see train()), so the checkpoint kept is the one whose SAMPLED actions
best match held-out demonstrations. As everywhere in this project, judge the policy by
closed-loop rollouts (fm_policy_node_room.py), not this number.
"""

import argparse
import math
import os
import sys

import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError:
    raise SystemExit(
        "PyTorch not found. Install the CPU build:\n"
        "    pip3 install torch --index-url "
        "https://download.pytorch.org/whl/cpu")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bc_train_room import load_dataset, to_img_tensor, write_curve  # noqa: E402
from fm_losses import (  # noqa: E402
    FlowMatchingSchedule, flow_matching_loss, sample_actions,
)


def time_embedding(t, dim):
    """t: (B,) in [0, 1] -> (B, dim) sinusoidal features."""
    half = dim // 2
    freqs = torch.exp(torch.linspace(0.0, math.log(1000.0), half,
                                     device=t.device))
    ang = t.unsqueeze(-1) * freqs * math.pi
    return torch.cat([ang.sin(), ang.cos()], dim=-1)


class FlowPolicy(nn.Module):
    """(camera frame, odometry, noisy action a_t, time t) -> velocity.

    Same convolutional encoder as bc_train_room.py's CNNPolicy; the head
    additionally takes the noisy action and a time embedding."""

    def __init__(self, odom_dim, img_size=64, chunk=1, img_embed=128,
                hidden=(128, 64), t_embed=32):
        super().__init__()
        self.chunk = chunk
        self.img_size = img_size
        self.t_embed = t_embed
        self.act_dim = 2 * chunk

        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.ReLU(),
        )
        conv_hw = img_size // 8
        self.img_fc = nn.Sequential(
            nn.Linear(32 * conv_hw * conv_hw, img_embed), nn.ReLU())

        layers, d = [], img_embed + odom_dim + self.act_dim + t_embed
        for h in hidden:
            layers += [nn.Linear(d, h), nn.SiLU()]
            d = h
        layers.append(nn.Linear(d, self.act_dim))
        self.head = nn.Sequential(*layers)

    def encode(self, img):
        return self.img_fc(self.conv(img).flatten(1))

    def velocity(self, z_img, odom, a_t, t):
        """Predicted velocity da/dt. z_img (B, img_embed) = encode(img),
        odom (B, odom_dim), a_t (B, *act_shape) noisy action, t (B,) in
        [0, 1]. -> same shape as a_t.

        Concatenate the image embedding, odometry, the flattened noisy
        action and time_embedding(t, self.t_embed), and map through
        self.head. Like forward(), but with the image already encoded -
        sampling calls this once per integration step."""
        B = a_t.shape[0]
        h = torch.cat([z_img, odom, a_t.reshape(B, -1),
                       time_embedding(t, self.t_embed)], dim=1)
        return self.head(h).view_as(a_t)

    def forward(self, img, odom, a_t, t):
        return self.velocity(self.encode(img), odom, a_t, t)

    def act_shape(self):
        return (self.chunk, 2) if self.chunk > 1 else (2,)

    @torch.no_grad()
    def sample(self, img, odom, steps=10, noise=None):
        """-> (B, *act_shape) normalised action(s), one per noise draw."""
        z = self.encode(img)
        return sample_actions(lambda _i, o, a, t: self.velocity(z, o, a, t),
                              None, odom, self.act_shape(), steps, noise)


def evaluate(model, schedule, img_u8, odom, Y, device, batch=256, seed=0):
    """Flow-matching loss on held-out data, with FIXED noise and t (seeded)
    so successive epochs are compared on the same draws."""
    model.eval()
    g = torch.random.get_rng_state()
    torch.manual_seed(seed)
    total, n = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(odom), batch):
            xb = to_img_tensor(img_u8[i:i + batch], device)
            ob, yb = odom[i:i + batch], Y[i:i + batch]
            total += flow_matching_loss(model, schedule, xb, ob, yb).item() \
                * len(ob)
            n += len(ob)
    torch.random.set_rng_state(g)
    return total / max(n, 1)


def sample_mae(model, img_u8, odom, Y, act_std, chunk, device, steps,
               batch=256):
    """Mean |sampled action - demonstrated action|, in real units. One
    sample per state, so where the data is multimodal this is LARGER than
    a regression head's error even for a good policy (a sample on the
    other mode counts as a big miss) - a health check only."""
    model.eval()
    errs, n = np.zeros(2, np.float64), 0
    for i in range(0, len(odom), batch):
        xb = to_img_tensor(img_u8[i:i + batch], device)
        ob = odom[i:i + batch]
        err = (model.sample(xb, ob, steps) - Y[i:i + batch]).abs()
        err = err.view(len(ob), -1, 2).mean(dim=1).cpu().numpy()
        errs += err.sum(axis=0)
        n += len(ob)
    return (errs / max(n, 1)) * act_std


def train(args):
    d, chunk, history, img_size = load_dataset(args.dataset)
    dev = torch.device('cuda' if (args.cuda and torch.cuda.is_available())
                       else 'cpu')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    Xtr_img = d['X_train_img']                 # keep as uint8 on host RAM
    Xva_img = d['X_val_img']
    Xtr_odom = torch.tensor(d['X_train_odom'], dtype=torch.float32, device=dev)
    Xva_odom = torch.tensor(d['X_val_odom'], dtype=torch.float32, device=dev)
    act_shape = (chunk, 2) if chunk > 1 else (2,)
    Ytr = torch.tensor(d['Y_train'], dtype=torch.float32,
                       device=dev).view(-1, *act_shape)
    Yva = torch.tensor(d['Y_val'], dtype=torch.float32,
                       device=dev).view(-1, *act_shape)

    odom_dim = Xtr_odom.shape[1]
    model = FlowPolicy(odom_dim, img_size=img_size, chunk=chunk,
                       img_embed=args.img_embed, hidden=tuple(args.hidden),
                       t_embed=args.t_embed).to(dev)
    schedule = FlowMatchingSchedule()

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)

    n = len(Xtr_odom)
    act_std = d['act_std']
    best, best_state, best_epoch = float('inf'), None, 0
    history_log = []

    print(f"\ndevice {dev} | odom {odom_dim} | img {img_size}x{img_size} | "
          f"chunk {chunk} | hidden {tuple(args.hidden)} | flow matching")
    print(f"train {n} samples, val {len(Xva_odom)}\n")
    print(f"{'epoch':>6} {'train':>10} {'val':>10} "
          f"{'val MAE v':>10} {'val MAE w':>10}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = np.random.permutation(n)
        running = 0.0
        for i in range(0, n, args.batch):
            idx = perm[i:i + args.batch]
            xb = to_img_tensor(Xtr_img[idx], dev)
            loss = flow_matching_loss(model, schedule, xb, Xtr_odom[idx],
                                      Ytr[idx])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running += loss.item() * len(idx)

        tr = running / n
        va = evaluate(model, schedule, Xva_img, Xva_odom, Yva, dev)
        history_log.append((epoch, tr, va))

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            # Checkpoint selection uses the error of SAMPLED actions on
            # held-out episodes, not the validation flow-matching loss:
            # with this little data the val loss starts rising within ~10
            # epochs (the velocity field gets large far from the training
            # frames) while the actions it actually produces keep
            # improving for much longer.
            mae = sample_mae(model, Xva_img, Xva_odom, Yva, act_std, chunk,
                             dev, args.sample_steps)
            score = float((mae / act_std).sum())
            mark = ""
            if score < best:
                best, best_epoch, mark = score, epoch, " *"
                best_state = {k: v.detach().clone()
                              for k, v in model.state_dict().items()}
            print(f"{epoch:6d} {tr:10.5f} {va:10.5f} "
                  f"{mae[0]:10.4f} {mae[1]:10.4f}{mark}")

    if best_state is not None:
        model.load_state_dict(best_state)
    mae = sample_mae(model, Xva_img, Xva_odom, Yva, act_std, chunk, dev,
                     args.sample_steps)

    torch.save(dict(
        kind='flow_matching',
        state_dict=model.state_dict(),
        odom_dim=odom_dim,
        img_size=img_size,
        chunk=chunk,
        history=history,
        img_embed=args.img_embed,
        hidden=list(args.hidden),
        t_embed=args.t_embed,
        sample_steps=args.sample_steps,
        odom_mean=d['odom_mean'], odom_std=d['odom_std'],
        act_mean=d['act_mean'], act_std=d['act_std'],
        best_epoch=best_epoch,
        val_mae_v=float(mae[0]), val_mae_w=float(mae[1]),
    ), args.out)

    print(f"\nkept epoch {best_epoch}")
    print(f"val MAE of one sample         v {mae[0]:.4f} m/s   "
          f"w {mae[1]:.4f} rad/s")
    print(f"                              (demonstrated v spread "
          f"{act_std[0]:.3f}, w spread {act_std[1]:.3f})")

    if args.curve:
        write_curve(history_log, args.curve)
        print(f"\nloss curve -> {args.curve}")

    print(f"\nsaved {args.out}")
    print("\nRun it in the simulator to find out what it actually does "
          "(fm_policy_node_room.py).\n")


def build_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dataset', default='dataset_room.npz')
    p.add_argument('--out', default='policy_room_fm.pt')
    p.add_argument('--img-embed', type=int, default=128)
    p.add_argument('--hidden', type=int, nargs='+', default=[128, 64])
    p.add_argument('--t-embed', type=int, default=32)
    p.add_argument('--sample-steps', type=int, default=10,
                   help='Euler steps from noise to action at test time')
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--batch', type=int, default=64)
    p.add_argument('--epochs', type=int, default=150)
    p.add_argument('--eval-every', type=int, default=10,
                   help='epochs between sampled-action validation checks')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--curve', default='figures/training_curve_room_fm.png')
    p.add_argument('--cuda', action='store_true')
    return p


if __name__ == '__main__':
    a = build_parser().parse_args()
    if a.curve:
        os.makedirs(os.path.dirname(a.curve) or '.', exist_ok=True)
    train(a)

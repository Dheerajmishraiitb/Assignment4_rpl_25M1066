# Assignment 4 — Learning to navigate a room: BC, DAgger, Flow Matching, Policy Gradient

**Course:** CS7001 — Robot Perception and Learning

One room, one TurtleBot3 (camera + odometry), one task: drive from a start
pose to the goal at the far end of the room, past the island in the middle
(either side is fine). You will get a policy for this task four different
ways and compare them.

Every part is graded by the same code: success and collisions are judged
from ground truth, which the policy itself never sees. The robot only gets
its camera image and odometry relative to where the trial started.

## Marks

| Part | Marks |
|---|---|
| 1 — Behavioural Cloning | 3 |
| 2 — DAgger | 2 |
| 3 — Flow Matching | 4 |
| 4 — Policy Gradient (REINFORCE) | 6 |
| **Total** | **15** |

Each part below lists what its marks are for. Everything you are asked to
report goes in **one Markdown file, `report.md`** - see
[Report](#report-reportmd) and [Submission](#submission).

## What is in this folder

```
README.md                       this handout
report.md                       your report - fill in the skeleton
gazebo_models/                  models the room world needs
src/room_nav_bc/                the ROS 2 package
  worlds/, config/, launch/     the environment, shared by every part
  config/start_goal.yaml        room layout, goal region, start poses (train/test/centre)
  scripts/record_demo_room.py   drive the robot with the keyboard, record demonstrations
  scripts/bc_dataset_room.py    episodes -> training set (.npz)
  scripts/bc_train_room.py      CNN + odometry policy, regression
  scripts/bc_policy_node_room.py  run a policy closed-loop and grade it
  scripts/dagger_room.py        policy drives, you correct it (Part 2)
  scripts/fm_losses.py          flow-matching math (Part 3)
  scripts/fm_train_room.py      flow-matching policy (Part 3)
  scripts/fm_policy_node_room.py  run/grade a flow-matching policy (Part 3)
  scripts/plot_policy_field_room.py  map of what BC and FM do all over the room (Part 3)
  scripts/pg_train_room.py      REINFORCE, online in Gazebo (Part 4)
  scripts/pg_policy_node_room.py  run/grade a REINFORCE policy (Part 4)
  scripts/expert_demo_room.py   scripted expert (used for its route geometry)
  scripts/show_starts_room.py   visualise the start poses in Gazebo
```

---

## Setup

You use the same course Docker container as in the earlier assignments.
**Edit files on your laptop; run every command inside the container.** The
assignment folder is bind-mounted, so both sides see the same files.

| do this | where |
|---|---|
| edit the scripts and `report.md` | **your laptop**, in VS Code |
| `colcon build`, `ros2 launch`, `ros2 run`, recording, training | **inside the container** |
| `git add` / `commit` / `push` | **your laptop** |

Gazebo appears in your **browser**, not as a desktop window.

**1. Clone the assignment into the container's workspace folder**

```bash
cd ~/CS7001-docker-setup/workspace
git clone https://github.com/cs7001-rpl-iitb/Assignment_4.git assignment_4
```

`CS7001-docker-setup/workspace` is the only folder the container mounts, at
`/home/ros/ws`. After cloning:

* on your laptop: `~/CS7001-docker-setup/workspace/assignment_4`
* inside the container: `/home/ros/ws/assignment_4`

**2. Start the container**

```bash
cd ~/CS7001-docker-setup
docker compose up -d
docker ps        # expect: ros2-humble-tb3   Up ... (healthy)
```

Open **<http://localhost:6080/vnc.html>**, password `turtlebot3`.

**3. Install PyTorch, inside the container**

Everything else this assignment needs (OpenCV, matplotlib, `cv_bridge`,
Gazebo 11, the camera TurtleBot) is already in the image.

```bash
docker exec -it ros2-humble-tb3 bash
pip3 install --user torch --index-url https://download.pytorch.org/whl/cpu
python3 -c "import torch; print(torch.__version__)"    # expect: 2.x.x+cpu
```

About 30 s and 850 MB. Use exactly this command: the container has no GPU,
and a plain `pip3 install torch` downloads several GB of CUDA libraries for
nothing. Do **not** `pip install` NumPy or `opencv-python` - `cv_bridge` needs
the NumPy 1.x the image ships with.

> ⚠️ **Stop the container, never `down` it.** PyTorch is installed *inside
> the container*, not in your mounted folder, so `docker compose down`
> deletes it. Use `docker compose stop` / `docker compose start`. If you do
> run `down` by accident, repeat the `pip3 install`. Your code, demos and
> models are in the mounted folder and survive either way.

**4. Build, inside the container**

```bash
cd /home/ros/ws/assignment_4
colcon build --symlink-install
source install/setup.bash
export GAZEBO_MODEL_PATH=$PWD/gazebo_models:$GAZEBO_MODEL_PATH
```

Expect `Summary: 1 package finished`. With `--symlink-install` a saved edit
takes effect immediately; re-run `colcon build` only if you add a new
script. **Every new terminal** needs the last three lines again - or add them
to the container's `~/.bashrc` once:

```bash
echo 'cd /home/ros/ws/assignment_4 && source install/setup.bash && export GAZEBO_MODEL_PATH=/home/ros/ws/assignment_4/gazebo_models:$GAZEBO_MODEL_PATH' >> ~/.bashrc
```

**5. Check it runs**

```bash
ros2 launch room_nav_bc room_demo.launch.py
```

The room and the robot appear in the browser tab after ~30-60 s. Add
`gui:=false` for a faster headless run (no window; use it for long
training and evaluation runs).

Open more terminals with `docker exec -it ros2-humble-tb3 bash`, or from the
browser desktop. Run every command from `/home/ros/ws/assignment_4`, so
data, `.pt` files, `figures/` and `logs/` land in one place on your laptop.

**How long things take** (course container, CPU only): BC training ~30 s,
flow-matching training ~2 min, the policy-field map ~1.5 min the first
time, REINFORCE ~3 h for 300 episodes (episodes run in real time). Part 4
needs at least two such runs - **start it early**. The camera runs at
~6-9 fps under the container's software rendering; that is expected.

**Grading rule used by Parts 1–3:** a trial succeeds if the robot is
**stopped** (|v| < 0.01 m/s and |w| < 0.05 rad/s for 1.5 s) **inside the
goal region**. Collisions are reported separately. Part 4 uses a different
rule — see there.

## What you implement

Look for `TODO` — each one is a function that raises `NotImplementedError`
with a docstring describing exactly what it must return.

| Part | File (in `src/room_nav_bc/scripts/`) | What |
|---|---|---|
| 1 | `bc_dataset_room.py` | `episode_arrays()` — observation and action per sample |
| 1 | `bc_train_room.py` | `CNNPolicy.forward()`, and one training step, `train_step()` |
| 3 | `fm_losses.py` | `interpolate()`, `target_velocity()`, `flow_matching_loss()`, `sample_actions()` |
| 3 | `fm_train_room.py` | `FlowPolicy.velocity()` |
| 4 | `pg_train_room.py` | `compute_reward()` — your reward design |

Self-checks that need no simulator (inside the container, from
`/home/ros/ws/assignment_4`):

```bash
python3 src/room_nav_bc/scripts/fm_losses.py    # Part 3 math, incl. a two-mode toy problem
python3 src/room_nav_bc/scripts/pg_losses.py    # REINFORCE math (given)
```

---

## Part 1 — Behavioural Cloning *[3 marks]*

1. **Record 5–10 demonstrations** from the `train` start poses. Drive to
   the goal and **stop inside it** before saving (ENTER). Take whichever
   side of the island feels natural each time.
   ```bash
   ros2 run room_nav_bc record_demo_room.py --data-dir data_room_human
   ```
2. **Build the dataset** (split by episode into train/validation):
   ```bash
   ros2 run room_nav_bc bc_dataset_room.py --data-dir data_room_human --out dataset_room_human.npz
   ```
3. **Train**:
   ```bash
   ros2 run room_nav_bc bc_train_room.py --dataset dataset_room_human.npz \
       --out policy_room_human.pt --curve figures/training_curve_room.png
   ```
4. **Test** on the unseen `test` start poses:
   ```bash
   ros2 run room_nav_bc bc_policy_node_room.py --policy policy_room_human.pt \
       --starts test --trials 3 --jitter-seed 0
   ```

**In `report.md`:** number of demonstrations, the training curve, and the
test table (success, collisions, what was hit).

**Marks:**
- 1.5 — the three TODOs work: the dataset builds, the training curve goes
  down, the policy drives in the simulator
- 0.5 — 5–10 demonstrations that end stopped inside the goal
- 1 — training curve and test table in the report, with a sentence on how
  the failed trials failed

## Part 2 — DAgger *[2 marks]*

The current policy drives; when it is about to go wrong, press SPACE, drive
it back on track, and press SPACE again to hand back control. Only the
steps you drove are added to the training set.

```bash
# round k: run the current policy and correct it
ros2 run room_nav_bc dagger_room.py --policy policy_room_human.pt \
    --starts train --trials 5 --round 1
# aggregate: your original demos + every round's corrections so far
ros2 run room_nav_bc bc_dataset_room.py --human-only \
    --data-dir data_room_human data_dagger/round_01 --out dataset_dagger_r1.npz
ros2 run room_nav_bc bc_train_room.py --dataset dataset_dagger_r1.npz --out policy_room_dagger_r1.pt
ros2 run room_nav_bc bc_policy_node_room.py --policy policy_room_dagger_r1.pt \
    --starts test --trials 3 --jitter-seed 0
```

Repeat for at least 2–3 rounds, each starting from the previous round's
policy, adding every round's folder to `--data-dir`.

**In `report.md` — one plot:** test success rate (y) against the number of
human-labelled samples in the training set (x), one point per policy:
Part 1's policy, then each DAgger round. Annotate each point with how many
of those samples came from DAgger corrections. The dataset builder prints
the sample counts. In a sentence or two: how much extra labelled data did
DAgger need to change the success rate, compared with what you recorded in
Part 1?

**Marks:**
- 1 — at least 2 DAgger rounds, each tested on the `test` starts
- 1 — the plot and the answer

## Part 3 — Flow Matching *[4 marks]*

Same dataset as Part 1, same CNN encoder. Instead of regressing one action
per observation, the policy learns a velocity field that carries random
noise onto the distribution of demonstrated actions; an action is drawn by
integrating it (`fm_losses.py` docstring has the math).

1. Implement the TODOs and pass `python3 src/room_nav_bc/scripts/fm_losses.py`.
2. Train on your Part 1 dataset:
   ```bash
   ros2 run room_nav_bc fm_train_room.py --dataset dataset_room_human.npz \
       --out policy_room_fm.pt --curve figures/training_curve_room_fm.png
   ```
3. Test it exactly like Part 1 — and **watch the runs in the Gazebo GUI**:
   ```bash
   ros2 run room_nav_bc fm_policy_node_room.py --policy policy_room_fm.pt \
       --starts test --trials 3 --jitter-seed 0
   ```
4. Map what both policies do all over the room. The first run needs the
   simulator (headless is fine): it teleports the robot to a grid of
   points and saves the camera frame at each one (about 2 min). Later runs
   reuse the saved frames and need no simulator.
   ```bash
   ros2 run room_nav_bc plot_policy_field_room.py --bc policy_room_human.pt \
       --fm policy_room_fm.pt --data-dir data_room_human --out figures/policy_field.png
   ```

**Questions (answer in `report.md`)**

- **Q3.1** Flow matching is the more expressive method — it can represent
  any action distribution, BC only a single action per observation. Did it
  get the same success rate as BC? If not, why not?
  *Hint: watch the runs and look at what the robot does once it reaches
  the goal, then re-read how success is defined above.*
- **Q3.2** Include `figures/policy_field.png`. At every grid point the
  robot faces the goal, and its odometry is measured from start A, as if
  a trial had begun there. Each arrow is where the robot would drive in
  the next 0.8 s; flow matching draws 8 sampled actions per point.
  (a) Compare the two policies along the grey demonstration paths and
  inside the goal box. How does the picture explain your answer to Q3.1?
  (b) Now look far from the demonstrations, e.g. along the side walls.
  Are those arrows sensible? What does this tell you about what a learned
  policy does in states that are not in its training data, and how does
  that relate to what DAgger fixes?

**Marks:**
- 1.5 — the five TODOs: `fm_losses.py` self-check passes, training runs
- 0.5 — test table on the `test` starts
- 1 — Q3.1
- 1 — Q3.2

## Part 4 — Policy Gradient (REINFORCE) *[6 marks]*

Part 3 showed that *how success is measured* decides what counts as a good
policy. In reinforcement learning you write that measure yourself: the
reward. Here there are no demonstrations — the policy (odometry only, no
camera) starts random and learns online in Gazebo from the reward you
design. The REINFORCE algorithm itself (`pg_losses.py`, `pg_networks.py`,
the training loop) is given; **your job is `compute_reward()`** in
`pg_train_room.py`.

Training uses one start pose (`centre`); an episode ends on collision,
entering the goal region, standing still, or timeout. Success here is
**entering** the goal region — no stop required.

```bash
# terminal 1 (headless is much faster)
ros2 launch room_nav_bc room_demo.launch.py gui:=false
# terminal 2 (300 episodes, ~3 h)
ros2 run room_nav_bc pg_train_room.py --out policy_pg.pt \
    --episode-log logs/pg_episodes.csv --curve figures/pg_training_curve.png
# afterwards (GUI on, to watch)
ros2 run room_nav_bc pg_policy_node_room.py --policy policy_pg.pt --trials 5
```

Train **at least two different reward designs** (one full run each), and
give each run its own `--out`, `--episode-log` and `--curve` names so the
second does not overwrite the first.

This is open-ended: plain REINFORCE with a few hundred episodes is a hard
setting, and a policy that reaches the goal is not required. What is
graded is your reasoning.

**Questions (answer in `report.md`)**

- **Q4.1** What reward did you design, and why? Describe every term and
  what behaviour it is meant to encourage or discourage.
- **Q4.2** Plot episode return against training episode (the script writes
  `figures/pg_training_curve.png` and the per-episode CSV). Describe what
  the policy learned at each stage of the curve, using what you saw it do.
- **Q4.3** Did the policy find a way to score well that is not what you
  intended? Describe at least one reward you tried, what the policy did
  with it, and what you changed.

**Marks:**
- 1 — `compute_reward()` implemented; at least two full training runs,
  with their curves and episode CSVs submitted
- 2 — Q4.1
- 1.5 — Q4.2
- 1.5 — Q4.3

---

## Report (`report.md`)

Write the report in **Markdown**, in the file `report.md` at the top of
this folder - a skeleton with one section per part is already there.

- Put every figure in `figures/` and embed it with a relative path, e.g.
  `![BC training curve](figures/training_curve_room.png)`, so it renders
  on GitHub and in VS Code's Markdown preview.
- Give test results as Markdown tables (copy the numbers from the test
  node's printout).
- Keep it short: the questions ask for explanations, not essays.



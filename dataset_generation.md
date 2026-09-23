# Dataset Generation: RGB → Depth Anything V2 pseudo-depth

How to generate training data for the RGB pipeline (`launch_evaluation_rgb.bash`), where
the policy flies on a monocular depth estimate instead of Flightmare's ground-truth depth.

## Why

`ViTLSTM_model.pth` was trained on Flightmare's **ground-truth** depth, but the RGB pipeline
feeds it **Depth Anything V2's estimate** of depth. Those are different distributions, so the
policy is being tested off-distribution — a train/test mismatch that's a likely contributor to
the poor `spheres_medium` results.

This pipeline closes that gap: collect RGB, run DA2 over it offline, save the estimate as the
training image, and retrain. The policy then trains on exactly what it will see at test time.

Because the DA2 output is converted into the *same* normalized encoding the original depth
dataset uses, **no model changes are needed** — the existing 1-channel models and
`training/dataloading.py` consume it unchanged.

## Pipeline

```
launch_evaluation.bash N state        expert policy flies, saves depth + rgb + telemetry
            |                          -> envtest/ros/train_set/<id>/
            v
make_da2_dataset.py                   DA2 over each rgb frame -> normalized pseudo-depth
            |                          -> training/datasets/data_da2/<id>/
            v
train.py                              unchanged training, dataset = data_da2
```

## 1. Collect

```bash
bash launch_evaluation.bash 20 state
```

Runs the privileged look-ahead expert (not a learned policy) at `real_time_factor:=10.0`.
Each trial writes one folder to `envtest/ros/train_set/<epoch_id>/`:

| File | Contents |
|---|---|
| `<timestamp>.png` | ground-truth depth, 480×640 uint8 grayscale |
| `<timestamp>_rgb.png` | rgb frame, 480×640×3 uint8, **BGR** order |
| `data.csv` | 21 columns of telemetry, one row per frame |

Frames are sampled every `self.time_interval = 0.03` s while `2 < pos_x < 60`.

The expert has a limited horizon and occasionally crashes; `data.csv`'s `is_collide` column
marks those frames. The dataloader does not filter them by default (the collision-skip block
in `dataloading.py` is commented out upstream).

## 2. Convert to pseudo-depth

```bash
python3 models/DepthAnythingV2/make_da2_dataset.py \
  --in envtest/ros/train_set \
  --out training/datasets/data_da2 \
  --compare
```

For each `<timestamp>_rgb.png`: run DA2 → metric meters → calibration curve → normalized
`[0,1]` → write `<timestamp>.png` as uint8. `data.csv` is copied verbatim.

`--compare` additionally reports mean absolute error against the paired ground-truth depth
frame, which is a cheap quality signal for the estimate.

Folders without `*_rgb.png` (e.g. anything collected in `vision` mode) are skipped and
reported, so it's safe to point `--in` at a `train_set` containing older runs.

**Output goes to a separate directory on purpose.** `dataloading.py` globs `*.png`
indiscriminately, so if the `_rgb.png` files sat alongside the training images it would load
them as depth frames, blow the image/telemetry count check, and silently skip every
trajectory.

## 3. Train

Set in `training/config/train.txt`:

```
dataset = data_da2
basedir = /root/catkin_ws/src/vitfly     # repo root; shipped value is the original author's path
```

`basedir` is joined with `datadir` and `logdir` (`train.py:93`, `158`, `69`), so it must point
at the repo root or both dataloading and logging fail. Inside the container that path is
`/root/catkin_ws/src/vitfly`.

```bash
python3 training/train.py --config training/config/train.txt
```

## Data format

The encoding all three stages agree on:

```
pixel_value / 255.0  ==  clip(raw_unity_depth / 0.09, 0, 1)
```

where `0.09` is `depth_im_threshold` in `run_competition.py`. Low values mean *close*
(danger), and anything beyond roughly 10 m saturates at 1.0. `dataloading.py` reads these with
`IMREAD_GRAYSCALE`, divides by 255, and resizes to 60×90 for the models.

DA2 emits metric meters, so `DepthEstimator.predict_normalized()` maps meters onto this
encoding via the calibration curve below.

## Calibration

`models/DepthAnythingV2/depth_calibration.npz` is a lookup table from DA2 meters to the
normalized target. Flightmare's exact depth-buffer encoding isn't available in this repo (the
Unity side is a compiled binary), so it's fit empirically rather than derived analytically.

Re-fit it offline from collected pairs — no simulator needed:

```bash
python3 models/DepthAnythingV2/calibrate_offline.py --in envtest/ros/train_set
```

It pairs each `_rgb.png` with its ground-truth `.png`, bins DA2's predicted meters, takes the
median target per bin, and enforces monotonicity (farther ⇒ higher target). `calibrate.py` is
the equivalent live-simulator version, kept for reference.

Re-fit when the scene type changes substantially, or after changing the DA2 checkpoint or
input preprocessing.

## Gotchas

These were real bugs hit while building this, fixed here but worth knowing if you pull fresh
upstream code:

- **`run_competition.py` wrote garbage RGB.** It did `(self.rgb_img*255).astype(np.uint8)` on
  an array that is *already* uint8 0-255, so it overflowed and saved the photographic
  negative. Note that a negative still looks like a plausible image, so eyeballing won't catch
  it — check that sky is bright and ground is dark.
- **The `unity/image` topic is `bgr8`.** A `passthrough` conversion is therefore already in
  OpenCV's BGR order, which is what DA2's `infer_image` expects. Flipping channels before
  handing it to DA2 gives it red/blue swapped. (The live pipeline did exactly this for a
  while.)
- **Calibration coverage depends on what the expert flew through.** An early fit sampled
  only frames where nothing was close, so its target bottomed out at 0.283 and the model
  received no near-field collision signal. Check the printed `target` range spans close to
  `0.0 → 1.0`.
- **Image/telemetry off-by-one** is normal: the last frame often has no telemetry row, and
  `dataloading.py` trims it.

## Status

Verified end to end on a 3-trajectory smoke test (609 frames): the dataloader consumed the
generated dataset with 0 skipped folders and 0 skipped images. Mean absolute error of the
pseudo-depth against ground-truth depth was **0.128** in normalized units.

That is a smoke test, not a training set — the original depth dataset is 580 trajectories.
No policy has been trained on DA2 pseudo-depth yet, so whether it actually improves RGB-pipeline
flight performance is still an open question.

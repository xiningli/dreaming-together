"""Stage 0 — visual pretraining (G1 second half).

1. Render ~50k segmentation frames from agent cameras across oracle-play
   states and randomized poses (all cameras, both teams — the encoder is
   shared).
2. Train SegEncoder + SegDecoder to reconstruct the class map from the
   256-d embedding (design v2: segmentation-only loss; the depth head was
   removed with the depth channel).
3. Gate: pixel accuracy > 90% on 5k held-out frames. Because background
   dominates, the log also reports non-background accuracy — the number
   that actually says whether tanks/shields survive the bottleneck.

Run: python -m dreaming_together.vision.pretrain_stage0 [--frames 50000]
Outputs: runs/stage0/encoder.pt, dataset cached at runs/stage0/frames.npy
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from dreaming_together.vision.encoder import SegEncoder, SegDecoder

RUN_DIR = ROOT / "runs" / "stage0"


def generate_frames(n_frames: int, seed: int = 0) -> np.ndarray:
    import mujoco
    from dreaming_together.envs.combat_env import CombatEnv, PREFIXES
    from dreaming_together.envs.cameras import SegCamera
    from dreaming_together.envs.tank import set_arm_ctrl, ARM_PAN_RANGE, ARM_TILT_RANGE
    from dreaming_together.oracle import ScriptedTeam

    rng = np.random.default_rng(seed)
    env = CombatEnv(seed=seed)
    cam = SegCamera(env.model)
    cameras = ["red0_front_cam", "red0_rear_cam", "red1_front_cam",
               "blue0_front_cam", "blue0_rear_cam", "blue1_front_cam"]
    frames = np.zeros((n_frames, 64, 64), dtype=np.int8)
    k = 0
    teams = (ScriptedTeam(0), ScriptedTeam(1))
    t0 = time.time()
    while k < n_frames:
        env.reset(seed=int(rng.integers(1 << 30)))
        # half the data: oracle play; other half: randomized arm poses
        randomize = rng.random() < 0.5
        steps = 0
        while not env.done and k < n_frames and steps < 120:
            if randomize:
                for p in PREFIXES:
                    set_arm_ctrl(env.model, env.data, p,
                                 rng.uniform(*ARM_PAN_RANGE),
                                 rng.uniform(*ARM_TILT_RANGE))
                import mujoco as mj
                mj.mj_forward(env.model, env.data)
                actions = {p: np.concatenate([rng.uniform(-1, 1, 2),
                                              rng.uniform(-1, 1, 2), [-1.0]])
                           for p in PREFIXES}
            else:
                actions = {}
                for tm in teams:
                    actions.update(tm.act(env))
            env.step(actions)
            steps += 1
            if steps % 4 == 0:   # sample every 200 ms
                for c in cameras:
                    if k >= n_frames:
                        break
                    frames[k] = cam.render(env.data, c)
                    k += 1
        if k and k % 5000 < len(cameras):
            rate = k / (time.time() - t0)
            print(f"  {k}/{n_frames} frames ({rate:.0f} f/s)", flush=True)
    cam.close()
    return frames


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=50_000)
    ap.add_argument("--epochs", type=int, default=15)
    args = ap.parse_args()
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    cache = RUN_DIR / "frames.npy"
    if cache.exists() and len(np.load(cache, mmap_mode="r")) >= args.frames:
        frames = np.load(cache)[:args.frames]
        print(f"loaded {len(frames)} cached frames")
    else:
        print(f"rendering {args.frames} frames ...")
        frames = generate_frames(args.frames)
        np.save(cache, frames)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    n_val = min(5000, len(frames) // 10)
    val, train = frames[:n_val], frames[n_val:]
    train_t = torch.from_numpy(train.astype(np.int64))
    val_t = torch.from_numpy(val.astype(np.int64)).to(dev)

    # class weights: background dominates; upweight rare classes
    counts = np.bincount(frames.reshape(-1).astype(np.int64), minlength=6)
    w = (counts.sum() / np.maximum(counts, 1)) ** 0.5
    weights = torch.tensor(w / w.sum() * 6, dtype=torch.float32, device=dev)
    print("class pixel counts:", counts.tolist())

    enc, dec = SegEncoder().to(dev), SegDecoder().to(dev)
    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()),
                           lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    t0 = time.time()
    for epoch in range(args.epochs):
        perm = torch.randperm(len(train_t))
        tot = 0.0
        for s in range(0, len(train_t), 256):
            mb = train_t[perm[s:s + 256]].to(dev)
            logits = dec(enc(mb))
            loss = loss_fn(logits, mb)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss) * len(mb)
        # validation
        with torch.no_grad():
            correct = nb_correct = nb_total = 0
            for s in range(0, len(val_t), 512):
                mb = val_t[s:s + 512]
                pred = dec(enc(mb)).argmax(1)
                correct += int((pred == mb).sum())
                nb = mb != 0
                nb_correct += int((pred[nb] == mb[nb]).sum())
                nb_total += int(nb.sum())
        acc = correct / val_t.numel()
        nb_acc = nb_correct / max(1, nb_total)
        print(f"epoch {epoch:2d}  loss {tot/len(train_t):.4f}  "
              f"pixel-acc {acc:.4f}  non-bg-acc {nb_acc:.4f}", flush=True)

    torch.save(enc.state_dict(), RUN_DIR / "encoder.pt")
    gate = acc > 0.90
    print(f"\nStage 0 gate ({'PASS' if gate else 'FAIL'}): "
          f"pixel-acc {acc:.4f} (need > 0.90), non-background {nb_acc:.4f}, "
          f"{time.time()-t0:.0f}s on {dev}")
    print(f"encoder → {RUN_DIR/'encoder.pt'}")
    return 0 if gate else 1


if __name__ == "__main__":
    raise SystemExit(main())

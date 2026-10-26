"""
Assignment 2: training pipeline (counterfeit peso screening).

    uv run python main.py train

Start to finish, no manual steps:
 1. Download the Roboflow Universe dataset (key via api_key.py).
 2. Describe it from the labels: images and instances per class and split,
    object size relative to the image.
 3. Finetune yolo11n from the pretrained COCO checkpoint; copy best weights
    to weights/best.pt.
 4. Evaluate on the TEST split: per-class precision/recall/AP and the
    confusion matrix. Baselines: COCO yolo11n zero-shot and predict-nothing.
 5. Score every validation and test note (highest "counterfeit" confidence).
 6. Sweep thresholds, cost them with the cost matrix in banknotes.py, choose
    the operating point on VALIDATION, report it on TEST.
 7. Save everything app.py needs into artifacts/.

Settings via environment variables: EPOCHS (default 50), MODEL (default
yolo11n.pt), IMGSZ (default 640), BATCH (default 16).

After changing a cost in banknotes.py, redo only step 6 (no retraining):
    COSTING_ONLY=1 uv run python train.py
"""

import json
import os
import shutil
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from ultralytics import YOLO

from banknotes import (ARTIFACTS, BASE_RATE, COST_FP, COST_REVIEW, COST_TN, COST_TP, DATASETS,
                       EUR_PER_PHP, PROJECT, ROOT, UNIVERSE_URL, VERSION, WEIGHTS, WORKSPACE,
                       cost_per_1000, is_positive, peso_value, staffing)

EPOCHS = int(os.environ.get("EPOCHS", "50"))
MODEL = os.environ.get("MODEL", "yolo11n.pt")
IMGSZ = int(os.environ.get("IMGSZ", "640"))
BATCH = int(os.environ.get("BATCH", "16"))
SEED = 42
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ARTIFACTS.mkdir(exist_ok=True)
WEIGHTS.mkdir(exist_ok=True)


def save_json(name, obj):
    (ARTIFACTS / name).write_text(json.dumps(obj, indent=2, default=float))


# ---------------------------------------------------------------- 1. download
def download_dataset() -> Path:
    target = DATASETS / f"{PROJECT}-v{VERSION}"
    if not (target / "data.yaml").exists():
        from roboflow import Roboflow
        from api_key import load_api_key
        rf = Roboflow(api_key=load_api_key())
        version = rf.workspace(WORKSPACE).project(PROJECT).version(VERSION)
        try:
            version.download("yolov11", location=str(target))
        except Exception:  # older SDKs only know the yolov8 name; same label format
            version.download("yolov8", location=str(target))
    # Roboflow writes relative paths that break depending on the working
    # directory; write a copy of data.yaml with absolute paths.
    cfg = yaml.safe_load((target / "data.yaml").read_text())
    for split, folder in [("train", "train"), ("val", "valid"), ("test", "test")]:
        cfg[split] = str((target / folder / "images").resolve())
    cfg.pop("path", None)
    (target / "data_abs.yaml").write_text(yaml.safe_dump(cfg))
    return target


def split_files(root: Path, folder: str):
    imgs = sorted(p for p in (root / folder / "images").iterdir() if p.suffix.lower() in IMG_EXT)
    return imgs, [root / folder / "labels" / (p.stem + ".txt") for p in imgs]


def read_labels(label_path: Path):
    if not label_path.exists():
        return []
    rows = []
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 5:
            cls = int(float(parts[0]))
            if len(parts) == 5:  # box: class x y w h (relative to the image)
                w, h = float(parts[3]), float(parts[4])
            else:                # polygon: use its bounding box
                xs, ys = [float(v) for v in parts[1::2]], [float(v) for v in parts[2::2]]
                w, h = max(xs) - min(xs), max(ys) - min(ys)
            rows.append((cls, w, h))
    return rows


# ---------------------------------------------------------------- 2. describe
def describe(root: Path, names: dict) -> dict:
    stats = {"classes": names, "splits": {}}
    for split in ["train", "valid", "test"]:
        imgs, labels = split_files(root, split)
        inst, img_with = Counter(), Counter()
        areas, n_empty, n_fake_img, n_multi = [], 0, 0, 0
        for lp in labels:
            rows = read_labels(lp)
            n_empty += not rows
            n_multi += len(rows) > 1
            present = set()
            for cls, w, h in rows:
                inst[names[cls]] += 1
                present.add(names[cls])
                areas.append(w * h)
            for n in present:
                img_with[n] += 1
            n_fake_img += any(is_positive(n) for n in present)
        a = np.array(areas) if areas else np.array([0.0])
        stats["splits"][split] = {
            "images": len(imgs), "images_without_labels": n_empty, "images_with_several_boxes": n_multi,
            "images_with_fake": n_fake_img, "share_images_with_fake": n_fake_img / max(len(imgs), 1),
            "instances_per_class": dict(sorted(inst.items())), "images_per_class": dict(sorted(img_with.items())),
            "box_area_share_median": float(np.median(a)), "box_area_share_p10": float(np.percentile(a, 10)),
            "box_area_share_p90": float(np.percentile(a, 90)),
            "share_small_boxes_lt_1pct": float((a < 0.01).mean()),
        }
    return stats


# ---------------------------------------------------------------- 5. scoring
def note_scores(model, root: Path, split: str, names: dict) -> pd.DataFrame:
    """One row per image (= one note): truth, face value and the highest fake score."""
    imgs, labels = split_files(root, split)
    pos_ids = [i for i, n in names.items() if is_positive(n)]
    rows = []
    for k in range(0, len(imgs), 32):
        batch = imgs[k:k + 32]
        results = model.predict([str(p) for p in batch], conf=0.001, imgsz=IMGSZ, verbose=False)
        for p, lp, r in zip(batch, labels[k:k + 32], results):
            cls = r.boxes.cls.cpu().numpy().astype(int)
            conf = r.boxes.conf.cpu().numpy()
            fake = conf[np.isin(cls, pos_ids)]
            real = conf[~np.isin(cls, pos_ids)]
            truth = read_labels(lp)
            fake_classes = [names[c] for c, _, _ in truth if c in pos_ids]
            all_classes = [names[c] for c, _, _ in truth]
            value_from = fake_classes or all_classes
            rows.append({"image": p.name, "is_fake": int(bool(fake_classes)),
                         "true_class": (fake_classes or all_classes or ["(none)"])[0],
                         "value_php": max([peso_value(n) for n in value_from] or [0]),
                         "fake_score": float(fake.max()) if fake.size else 0.0,
                         "real_score": float(real.max()) if real.size else 0.0})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 6. costing
GRID = np.round(np.arange(0.0, 1.0001, 0.01), 2)


def cost(df, low, high, **kw):
    return cost_per_1000(df.fake_score, df.is_fake, df.value_php, low, high, **kw)


def single_sweep(df):
    return pd.DataFrame([{"threshold": t, **cost(df, t, t)} for t in GRID])


def best_band(df):
    best = None
    for lo in GRID:
        for hi in GRID[GRID >= lo]:
            r = cost(df, lo, hi)
            if best is None or r["cost_per_1000"] < best[2]["cost_per_1000"] - 1e-9:
                best = (float(lo), float(hi), r)
    return best


def run_costing(val_df, test_df):
    """Cost sweeps, chosen thresholds (on VALIDATION), policies (on TEST), sensitivity."""
    val_sweep, test_sweep = single_sweep(val_df), single_sweep(test_df)
    val_sweep.to_csv(ARTIFACTS / "cost_sweep_val.csv", index=False)
    test_sweep.to_csv(ARTIFACTS / "cost_sweep_test.csv", index=False)
    t_star = float(val_sweep.loc[val_sweep.cost_per_1000.idxmin(), "threshold"])
    lo, hi, _ = best_band(val_df)

    def on_test(low, high):
        r = cost(test_df, low, high)
        return {"low": low, "high": high, **r, **staffing(r["review_share"])}

    policies = {
        "do_nothing_pass_all": on_test(1.01, 1.01),
        "person_checks_everything": on_test(0.0, 1.01),
        "model_alone": on_test(t_star, t_star),
        "model_plus_review": on_test(lo, hi),
    }

    # How the answer moves if the base rate or the value lost on a missed fake were different
    sens = []
    for br in [0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.50]:
        for scale in [0.5, 1.0, 2.0]:
            singles = [(t, cost(val_df, t, t, base_rate=br, fn_scale=scale)["cost_per_1000"]) for t in GRID]
            ts, cs = min(singles, key=lambda x: x[1])
            sens.append({"base_rate": br, "fn_cost_multiplier": scale, "best_single_threshold": float(ts),
                         "model_alone_cost": cs,
                         "do_nothing_cost": cost(val_df, 1.01, 1.01, base_rate=br, fn_scale=scale)["cost_per_1000"],
                         "check_everything_cost": cost(val_df, 0.0, 1.01, base_rate=br, fn_scale=scale)["cost_per_1000"]})
    pd.DataFrame(sens).to_csv(ARTIFACTS / "sensitivity.csv", index=False)

    return {
        "costs": {"fn": "face value of the fake (pesos x EUR_PER_PHP)", "eur_per_php": EUR_PER_PHP,
                  "fp": COST_FP, "tp": COST_TP, "tn": COST_TN, "review": COST_REVIEW, "base_rate": BASE_RATE},
        "chosen": {"single_threshold": t_star, "review_low": lo, "review_high": hi,
                   "chosen_on": "validation split"},
        "policies_on_test": policies,
    }


def print_policies(summary):
    cols = ["cost_per_1000", "review_share", "fake_caught", "genuine_failed", "reviews_per_day", "tellers"]
    print(pd.DataFrame(summary["policies_on_test"]).T[cols].round(3))


def costing_only():
    """Re-run only the costing after changing a cost in banknotes.py (no download, no training)."""
    val_df = pd.read_csv(ARTIFACTS / "val_scores.csv")
    test_df = pd.read_csv(ARTIFACTS / "test_scores.csv")
    summary = json.loads((ARTIFACTS / "metrics.json").read_text())
    summary.update(run_costing(val_df, test_df))
    save_json("metrics.json", summary)
    print(json.dumps(summary["chosen"], indent=1))
    print_policies(summary)


def main():
    t0 = time.time()
    root = download_dataset()
    data_yaml = root / "data_abs.yaml"
    names = dict(enumerate(yaml.safe_load(data_yaml.read_text())["names"]))
    print("classes:", names)

    stats = describe(root, names)
    save_json("dataset_stats.json", stats)
    print(json.dumps(stats["splits"], indent=1))

    # ---- 3. finetune from the pretrained COCO checkpoint
    model = YOLO(MODEL)
    model.train(data=str(data_yaml), epochs=EPOCHS, imgsz=IMGSZ, batch=BATCH, seed=SEED,
                patience=15, project=str(ROOT / "runs"), name="finetune", exist_ok=True, plots=True)
    best = Path(model.trainer.best)
    shutil.copy(best, WEIGHTS / "best.pt")
    if (best.parent.parent / "results.csv").exists():
        shutil.copy(best.parent.parent / "results.csv", ARTIFACTS / "training_curve.csv")
    tuned = YOLO(str(WEIGHTS / "best.pt"))

    # ---- 4. evaluate on TEST
    m = tuned.val(data=str(data_yaml), split="test", imgsz=IMGSZ, project=str(ROOT / "runs"),
                  name="test_eval", exist_ok=True, plots=True)
    per_class = []
    test_inst = stats["splits"]["test"]["instances_per_class"]
    for i, cls_idx in enumerate(m.box.ap_class_index):
        n = names[int(cls_idx)]
        per_class.append({"class": n, "positive": is_positive(n), "test_instances": test_inst.get(n, 0),
                          "precision": float(m.box.p[i]), "recall": float(m.box.r[i]),
                          "AP50": float(m.box.ap50[i]), "AP50_95": float(m.box.ap[i])})
    for png in ["BoxPR_curve.png", "PR_curve.png"]:
        if (Path(m.save_dir) / png).exists():
            shutil.copy(Path(m.save_dir) / png, ARTIFACTS / png)
    # AP needs every low-confidence box, but a confusion matrix at confidence
    # 0.001 is swamped by junk boxes, so recompute it at 0.25.
    m25 = tuned.val(data=str(data_yaml), split="test", imgsz=IMGSZ, conf=0.25, project=str(ROOT / "runs"),
                    name="test_eval_conf25", exist_ok=True, plots=True)
    cm = m25.confusion_matrix.matrix.tolist()  # rows = predicted, cols = true; last = background
    for png in ["confusion_matrix.png", "confusion_matrix_normalized.png"]:
        if (Path(m25.save_dir) / png).exists():
            shutil.copy(Path(m25.save_dir) / png, ARTIFACTS / png)

    # ---- baselines on TEST
    test_imgs, _ = split_files(root, "test")
    coco = YOLO(MODEL)  # the same checkpoint we started from, never finetuned
    coco_seen, coco_any = Counter(), 0
    for k in range(0, len(test_imgs), 32):
        for r in coco.predict([str(p) for p in test_imgs[k:k + 32]], conf=0.25, imgsz=IMGSZ, verbose=False):
            found = r.boxes.cls.cpu().numpy().astype(int)
            coco_any += int(len(found) > 0)
            for c in found:
                coco_seen[coco.names[int(c)]] += 1
    baselines = {
        "coco_zero_shot": {
            "AP50": 0.0, "AP50_95": 0.0,
            "note": "COCO's 80 classes contain no banknote, genuine or counterfeit class, so it can never "
                    "output our classes: AP is 0 by construction and every note would be passed.",
            "what_it_detects_on_test_conf_0.25": dict(coco_seen.most_common(10)),
            "images_with_any_detection": coco_any, "test_images": len(test_imgs),
        },
        "predict_nothing": {"AP50": 0.0, "AP50_95": 0.0,
                            "note": "No boxes at all: every note passes, every fake is accepted."},
    }

    # ---- 5. note-level scores
    val_df = note_scores(tuned, root, "valid", names)
    test_df = note_scores(tuned, root, "test", names)
    val_df.to_csv(ARTIFACTS / "val_scores.csv", index=False)
    test_df.to_csv(ARTIFACTS / "test_scores.csv", index=False)

    summary = {
        "dataset": {"workspace": WORKSPACE, "project": PROJECT, "version": VERSION, "url": UNIVERSE_URL},
        "training": {"model": MODEL, "epochs_requested": EPOCHS, "imgsz": IMGSZ, "batch": BATCH,
                     "seed": SEED, "patience": 15, "weights": "weights/best.pt",
                     "device": str(getattr(model.trainer, "device", "unknown")),
                     "epochs_run": int(getattr(model.trainer, "epoch", -1)) + 1},
        "classes": names,
        "test_overall": {"mAP50": float(m.box.map50), "mAP50_95": float(m.box.map),
                         "precision": float(m.box.mp), "recall": float(m.box.mr)},
        "test_per_class": per_class,
        "test_confusion_matrix": {"matrix": cm, "labels": list(names.values()) + ["background"],
                                  "orientation": "rows = predicted, columns = true", "conf": 0.25},
        "baselines": baselines,
        **run_costing(val_df, test_df),   # ---- 6. costing
    }
    summary["training"]["minutes"] = round((time.time() - t0) / 60, 1)
    save_json("metrics.json", summary)
    print(json.dumps({k: summary[k] for k in ["test_overall", "chosen"]}, indent=1))
    print_policies(summary)
    print(f"done in {summary['training']['minutes']} min")


if __name__ == "__main__":
    costing_only() if os.environ.get("COSTING_ONLY") else main()

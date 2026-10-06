"""TensorBoard event files -> CSV, and per-run M1/M2/M3 table (plan §5).

  python scripts/export_tb_to_csv.py --runs-dir <ROOT>/runs --out results/csv
"""
import argparse
import glob
import json
import os

import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

TAGS = ["reward_100", "reward", "loss", "q_values_mean", "speed_fps", "epsilon", "temperature",
        "per_beta", "td_error_mean"]


def export_run(run_dir, out_dir):
    name = os.path.basename(run_dir)
    ea = EventAccumulator(os.path.join(run_dir, "tb"), size_guidance={"scalars": 0})
    ea.Reload()
    for tag in TAGS + [t for t in ea.Tags()["scalars"] if t.startswith("sigma_snr")]:
        if tag in ea.Tags()["scalars"]:
            df = pd.DataFrame([(e.step, e.wall_time, e.value) for e in ea.Scalars(tag)],
                              columns=["frame", "wall_time", "value"])
            df.to_csv(os.path.join(out_dir, f"{name}__{tag.replace('/', '_')}.csv"), index=False)
    mpath = os.path.join(run_dir, "metrics.json")
    return {"run": name, **(json.load(open(mpath)) if os.path.exists(mpath) else {})}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", required=True)
    ap.add_argument("--out", default="results/csv")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows = [export_run(d, a.out) for d in sorted(glob.glob(os.path.join(a.runs_dir, "*"))) if os.path.isdir(d)]
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.out, "run_metrics.csv"), index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()

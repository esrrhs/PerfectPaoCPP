#!/usr/bin/env python3
"""Plot evaluation curves (WP / ADP vs the RuleAgent baseline) from the CSV
appended by perfectpao_train / perfectpao_eval.

Usage:
  python tools/plot_eval.py --csv ckpt/eval_vs_rule.csv --out ckpt/eval_curve.png
  python tools/plot_eval.py --csv a/eval_vs_rule.csv b/eval_vs_rule.csv \
      --labels "run A" "run B" --out ab.png          # A/B comparison
  python tools/plot_eval.py --csv ckpt/eval_vs_rule.csv --out c.png --watch 30
"""
import argparse
import os
import sys
import time

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    print("matplotlib is required: pip install matplotlib", file=sys.stderr)
    sys.exit(1)


def load_csv(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline().strip().split(",")
        idx = {name: i for i, name in enumerate(header)}
        for line in f:
            parts = line.strip().split(",")
            if len(parts) != len(header):
                continue
            try:
                rows.append((int(parts[idx["update"]]),
                             float(parts[idx["wp"]]),
                             float(parts[idx["adp"]])))
            except (KeyError, ValueError):
                continue
    rows.sort(key=lambda r: r[0])
    return rows


def render(paths, labels, out):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    for path, label in zip(paths, labels):
        rows = load_csv(path)
        if not rows:
            print(f"warning: no data rows in {path}", file=sys.stderr)
            continue
        upd = [r[0] for r in rows]
        wp = [r[1] for r in rows]
        adp = [r[2] for r in rows]
        ax1.plot(upd, wp, marker="o", markersize=3, label=label)
        ax2.plot(upd, adp, marker="o", markersize=3, label=label)
    ax1.axhline(1.0 / 3.0, color="gray", linestyle="--", linewidth=1,
                label="random chance (1/3)")
    ax1.set_ylabel("win rate vs RuleAgent")
    ax1.set_ylim(0, 1)
    ax1.grid(alpha=0.3)
    ax1.legend()
    ax2.set_ylabel("mean payoff (ADP)")
    ax2.set_xlabel("update")
    ax2.grid(alpha=0.3)
    ax2.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", nargs="+", required=True)
    ap.add_argument("--labels", nargs="+", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--watch", type=int, default=0,
                    help="re-render every N seconds until interrupted")
    args = ap.parse_args()

    labels = args.labels if args.labels else [os.path.basename(p) for p in args.csv]
    if len(labels) != len(args.csv):
        ap.error("--labels count must match --csv count")

    while True:
        render(args.csv, labels, args.out)
        if args.watch <= 0:
            break
        try:
            time.sleep(args.watch)
        except KeyboardInterrupt:
            break


if __name__ == "__main__":
    main()

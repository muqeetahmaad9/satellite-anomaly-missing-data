"""Exploration of ESA Mission 1 benchmark channels (41-46): sampling, real gaps, labelled events.

A gap is an interval between consecutive samples longer than GAP_FACTOR x the channel's
median sampling interval. Outputs results/tables/esa_channels.csv, esa_gaps.csv, esa_events.csv
and results/esa_explore_report.txt.
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from esa_data import load_channel, load_channel_info, load_labels

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results" / "tables"
CHANNELS = [f"channel_{i}" for i in range(41, 47)]
GAP_FACTOR = 10
SPLIT = pd.Timestamp("2007-01-01", tz="UTC")  # ESA-ADB: first half of the mission for training


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    lines = []
    log = lambda s="": (print(s), lines.append(str(s)))
    info = load_channel_info().set_index("Channel")
    labels = load_labels()

    rows, gaps = [], []
    for ch in CHANNELS:
        s = load_channel(ch)
        t = s.index.asi8 / 1e9
        dt = np.diff(t)
        med = float(np.median(dt))
        is_gap = dt > GAP_FACTOR * med
        g = dt[is_gap]
        rows.append({"channel": ch, "subsystem": info.loc[ch, "Subsystem"], "target": info.loc[ch, "Target"],
                     "samples": len(s), "start": s.index[0].date(), "end": s.index[-1].date(),
                     "median_dt_s": med, "p99_dt_s": float(np.percentile(dt, 99)),
                     "n_gaps": int(is_gap.sum()), "gap_hours_total": g.sum() / 3600,
                     "longest_gap_h": (g.max() / 3600) if len(g) else 0.0,
                     "value_min": float(s.min()), "value_max": float(s.max()),
                     "n_unique_values": int(s.nunique())})
        gaps.append(pd.DataFrame({"channel": ch, "start": s.index[:-1][is_gap], "gap_s": g}))
    ch_df = pd.DataFrame(rows).set_index("channel")
    ch_df.to_csv(TABLES / "esa_channels.csv")
    gap_df = pd.concat(gaps, ignore_index=True)
    gap_df.to_csv(TABLES / "esa_gaps.csv", index=False)

    with pd.option_context("display.width", 250, "display.max_columns", 30):
        log("=== Channels 41-46 ===")
        log(ch_df.round(2).to_string())

        log(f"\n=== Real gaps (> {GAP_FACTOR} x median interval) ===")
        q = gap_df.gap_s.describe(percentiles=[.5, .9, .99]) / 60
        log("gap length in minutes: " + ", ".join(f"{k} {v:.1f}" for k, v in q.items() if k != "count"))
        log(f"total gaps across channels: {len(gap_df):,}")
        per_year = gap_df.assign(year=gap_df.start.dt.year).groupby("year").size()
        log("gaps per year (all six channels): " + ", ".join(f"{y}: {n}" for y, n in per_year.items()))

        ev = labels[labels.Channel.isin(CHANNELS)].copy()
        ev["hours"] = (ev.EndTime - ev.StartTime).dt.total_seconds() / 3600
        ev["split"] = np.where(ev.StartTime < SPLIT, "train (2000-2006)", "test (2007-2013)")
        ev.to_csv(TABLES / "esa_events.csv", index=False)
        log("\n=== Labelled events on channels 41-46 ===")
        log(f"label rows: {len(ev)}, distinct events: {ev.ID.nunique()}")
        log(ev.groupby(["Category", "split"]).ID.nunique().unstack(fill_value=0).to_string())
        log("event duration (hours) by category:")
        log(ev.groupby("Category").hours.describe(percentiles=[.5]).round(2).to_string())
        comm = labels[labels.Category == "Communication Gap"]
        log(f"\ncommunication-gap events in the whole mission: {comm.ID.nunique()} "
            f"(channels affected per event: {comm.groupby('ID').Channel.nunique().tolist()})")

    (ROOT / "results" / "esa_explore_report.txt").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()

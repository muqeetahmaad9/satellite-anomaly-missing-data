"""Pilot follow-up: are block-removal false alarms driven by gap-related features,
and are they made with high confidence (so confidence-based rejection would miss them)?"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import BENCHMARK_FEATURES, build_table, load_segments
from pilot import SEED, clean, degraded_features, make_models, log, LINES, ROOT, TABLES

GAP_FEATURES = ["gaps_squared", "duration", "var_div_duration", "len", "len_weighted", "var_div_len"]


def main():
    segments = load_segments()
    df = build_table(segments)
    tr, te = df[df.train == 1], df[df.train == 0]
    yte = te.anomaly.to_numpy()
    normal = yte == 0
    sets = {"all 18": BENCHMARK_FEATURES,
            "no gaps_squared": [f for f in BENCHMARK_FEATURES if f != "gaps_squared"],
            "no timing/length": [f for f in BENCHMARK_FEATURES if f not in GAP_FEATURES]}
    rows = []
    for kind, rate in (("clean", 0.0), ("block", 0.2), ("random", 0.3)):
        Xd_all = clean(te[BENCHMARK_FEATURES]) if rate == 0 else degraded_features(segments, te.index, kind, rate, SEED)
        for sname, feats in sets.items():
            for mname, m in make_models().items():
                m.fit(clean(tr[feats]), tr.anomaly)
                p = m.predict_proba(Xd_all[feats])[:, 1]
                fa = normal & (p >= 0.5)
                conf = np.maximum(p, 1 - p)
                thr80 = np.quantile(conf, 0.2)  # accept the 80% most confident
                rows.append({"degradation": f"{kind} {int(rate*100)}%", "features": sname, "model": mname,
                             "FPR_normal": fa[normal].mean(),
                             "median_conf_false_alarms": np.median(conf[fa]) if fa.any() else np.nan,
                             "false_alarms_kept_at_80pct_coverage": (fa & (conf >= thr80)).sum() / max(fa.sum(), 1)})
    out = pd.DataFrame(rows).round(3)
    log("\n=== Pilot check: feature ablation and confidence of false alarms ===")
    log(out.to_string(index=False))
    out.to_csv(TABLES / "pilot_check.csv", index=False)
    (ROOT / "results" / "pilot_check_report.txt").write_text("\n".join(LINES), encoding="utf-8")


if __name__ == "__main__":
    main()

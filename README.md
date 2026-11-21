# Satellite telemetry anomaly detectors under missing data

Code for the paper *How Reliable Are Satellite Telemetry Anomaly Detectors Under Missing and Irregular Data?
A Selective-Prediction Study on OPS-SAT* (Muqeet Ahmad, Hizbullah Jadoon).

The study removes 5–30% of the observations in each telemetry segment, either as scattered points or as
one contiguous block, and measures how this changes:
- the detectors' false alarms, recall and calibration;
- the automate-or-defer decisions of reject-option rules (confidence threshold, conformal, gap-matched
  conformal and a learned error predictor).

It also tests gap-augmented training as a remedy and repeats the core experiments on ESA Mission 1.

## Setup

Python 3.11.

```
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

## Data

| Dataset | Source | Place it in |
| --- | --- | --- |
| OPSSAT-AD | Zenodo [15108715](https://zenodo.org/records/15108715) (CC BY 4.0): `dataset.csv`, `segments.csv` | `data/raw/` |
| ESA Anomalies Dataset, Mission 1 | Zenodo [15237121](https://zenodo.org/records/15237121): `ESA-Mission1.zip` | `data/esa/`, or set the `ESA_ROOT` environment variable |

To download Mission 1, run `python src/download_esa.py Mission1 data/esa 16`. The download is parallel,
resumable and MD5-checked. After downloading, extract the CSV files of the zip into `data/esa/ESA-Mission1/`.
`channels.csv` uses Deflate64 compression, which Python's `zipfile` cannot read, so extract it with 7-Zip or the
Windows file explorer.

The ESA channel files are pandas pickles. `src/esa_data.py` opens them only through a restricted unpickler that
accepts pandas and numpy classes and nothing else.

## Pipeline

Run the scripts from the repository root, in this order. Outputs go to `results/`.

| Step | Script | Main output |
| --- | --- | --- |
| Data audit | `src/audit.py` | `results/audit_report.txt` |
| Feature re-implementation check | `src/features.py` | `results/tables/features_recomputed.csv` |
| Pilot | `src/pilot.py`, `src/pilot_check.py` | `results/pilot_*` |
| Baselines (6 detectors × 5 seeds) | `src/run_baselines.py` | `results/tables/baselines_*.csv` |
| Stress test (scattered / block / jitter) | `src/stress_test.py` | `results/tables/stress_*.csv` |
| Models with a held-out calibration set | `src/run_calibrated.py [models] [--seeds 2,3]` | `results/models_cal/` |
| Decision rules (MaxProb, conformal, gap-matched / mismatched) | `src/uncertainty.py`, `src/plot_uncertainty.py` | `results/tables/uncertainty_*.csv` |
| Gap-augmented models (`<model>-Aug`, `<model>-AugR`) | `src/run_calibrated.py RandomForest-Aug ...`, then `src/uncertainty.py _aug1 <models>` | `results/tables/uncertainty_*_aug*.csv` |
| Learned error predictor | `src/error_predictor.py` | `results/tables/errpred_*.csv` |
| Equal-automation comparison | `src/selective_curves.py` | `results/tables/selective_curves*.csv` |
| Bootstrap CIs, calibration size, evidence deletion | `src/stats.py`, `src/stats.py deletion` | `results/tables/stats_*.csv` |
| Paper figures and tables | `src/paper_figures.py` | `results/paper/` |
| ESA: exploration and real outages | `src/esa_data.py`, `src/esa_explore.py` | `results/tables/esa_channels.csv`, `esa_gaps.csv`, `esa_events.csv` |
| ESA: one-hour windows | `src/esa_windows.py` | `$ESA_ROOT/windows.npz` |
| ESA: experiments (RF, XGBoost, 1D-CNN × 3 seeds) | `src/esa_experiment.py` | `results/tables/esa_runs.csv`, `esa_summary.csv` |
| ESA: low-drift years 2007–2008 | `src/esa_stable_period.py` | `results/tables/esa_stable*.csv`, `esa_fpr_by_year.csv` |

Core modules:
- `degrade.py`: controlled observation loss.
- `features.py`: the 18 benchmark features and the observation-quality features.
- `models.py`: the detectors.
- `metrics.py`: ECE and risk–coverage.
- `evidence.py`: evidence-preserving evaluation, i.e. whether a gap deleted the anomaly itself.

## Results

`results/tables/` contains every result table reported in the paper, as produced by the scripts above.
Trained models and per-segment predictions are not included because of their size; the scripts regenerate them.

## Key findings

- On OPSSAT-AD, 20% block loss raises the false-alarm rate of feature-based detectors from about 1% to 55–72%
  (+57.1 percentage points, 95% CI 53.4–60.7). Standard conformal prediction keeps only 29% of normal segments
  in its prediction set against a 95% target.
- Neural detectors on resampled signals stay reliable once anomalies whose evidence the gap deleted are
  accounted for.
- Gap-matched conformal calibration restores validity. Combined with gap-augmented training, it lets a random
  forest automate 75% of decisions at 5% error.
- ESA Mission 1 shows the same effects at a smaller size, together with strong temporal drift in 2009–2012.

## Citation

Please cite the paper once it is published (details will be added here).

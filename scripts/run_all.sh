#!/usr/bin/env bash
# Reproduce every result of the manuscript (about 3 to 4 hours on a 2-core CPU).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs results/raw data/processed
python experiments/e5_data_synth.py          | tee logs/e5.log   # real-data facts, synthetic fidelity
python experiments/e1_forecast.py            | tee logs/e1.log   # forecasting + estimator ensemble
python experiments/e0_crossfit_estimators.py | tee logs/e0.log   # cross-fitted estimators (ablation)
python experiments/e2_voi.py                 | tee logs/e2.log   # measured value of freshness
python experiments/e3_slow.py                | tee logs/e3.log   # main multi-agent comparison
python experiments/e4_policy_analysis.py     | tee logs/e4.log   # generalisation, slot budget, link margin, timing
python experiments/e6_ran.py                 | tee logs/e6.log   # RAN slice experiments
python scripts/make_outputs.py                                   # tables, figures, macros
cd manuscript && pdflatex main && bibtex main && pdflatex main && pdflatex main

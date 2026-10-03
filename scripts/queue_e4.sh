#!/bin/bash
# wait for the main comparison to finish, then run the policy analysis
cd "$(dirname "$0")/.."
while pgrep -f "experiments/e3_slow.py" > /dev/null; do sleep 30; done
python3 experiments/e4_policy_analysis.py > logs/e4.log 2>&1

#!/bin/bash
# Runs the experiments of DP.10, 3 runs of each case, with build/delivery_sim (or the binary you pass).
# usage (from the repository root):  tools/run_experiments.sh [binary]
# Raw files go to runs/<case>_r<N>.{log,out,err,summary}; then run tools/summarize_runs.py.
bin=${1:-build/delivery_sim}
reps=3
mkdir -p runs

one() {   # one <case> <scenario> [VAR=value ...]
  local name=$1 scenario=$2
  shift 2
  for r in $(seq 1 $reps); do
    env "$@" tools/run_scenario.sh "$bin" "config/scenarios/$scenario.json" "${name}_r$r" 0 | grep -v "^dispatcher:"
  done
}

# E1: how the estimates run. Without and with one slow estimate (500 ms) per order.
one e1_deferred fast EQ_ESTIMATES=deferred
one e1_async fast EQ_ESTIMATES=async
one e1_pool fast EQ_ESTIMATES=pool
one e1_slow_async fast EQ_ESTIMATES=async EQ_SLOW_ESTIMATE_MS=500
one e1_slow_pool fast EQ_ESTIMATES=pool EQ_SLOW_ESTIMATE_MS=500

# E2: hold the pickup slot while the food cooks, or wait for the food before taking a slot.
one e2_holding one_slot EQ_FOOD_WAIT=holding
one e2_before one_slot EQ_FOOD_WAIT=before

# E3: how many couriers are compared for one order.
for k in 1 2 4 8 16; do
  one e3_k$k fast EQ_CANDIDATES=$k
done

# E4: shortest paths at start-up on a 40x40 grid (1600 nodes), by number of tasks and by chunking.
python3 tools/make_grid_config.py 40 config/scenarios/grid40.json
for t in 1 2 4 8 16; do
  one e4_tasks$t grid40 EQ_GRAPH_TASKS=$t
done
one e4_static8 grid40 EQ_GRAPH_TASKS=8 EQ_GRAPH_GRAIN=0

#!/bin/bash
# Run one scenario, stop it with a signal, check the log and report races and stop time.
# usage: tools/run_scenario.sh <binary> <config.json> <name> <seconds-after-start|0> [INT|TERM]
#   seconds = 0 : wait for the run to end by itself (durationS)
#   example     : tools/run_scenario.sh build-tsan/delivery_sim config/scenarios/stress.json stress 5 INT
# Output files go to runs/<name>.{log,out,err,summary}.
bin=$1; cfg=$2; name=$3; after=${4:-0}; sig=${5:-INT}
mkdir -p runs
"$bin" "$cfg" --log "runs/$name.log" > "runs/$name.out" 2> "runs/$name.err" &
pid=$!
until grep -q "Simulation has started" "runs/$name.out" 2>/dev/null || ! kill -0 $pid 2>/dev/null; do sleep 0.01; done
if [ "$after" != "0" ]; then
  sleep "$after"
  t0=$(date +%s.%N); kill -$sig $pid; wait $pid; code=$?; t1=$(date +%s.%N)
  stop="stop took $(echo "$t1 - $t0" | bc) s"
else
  wait $pid; code=$?; stop="ended by itself"
fi
tail -n 1 "runs/$name.out" > "runs/$name.summary"
echo "$name: exit=$code, $stop, TSan warnings=$(grep -c 'WARNING: ThreadSanitizer' "runs/$name.err")"
grep "^dispatcher:" "runs/$name.err"
python3 tools/check_log.py "runs/$name.log" "runs/$name.summary" "$cfg"
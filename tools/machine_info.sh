#!/bin/bash
# Print the hardware and software facts the EC2 report needs (section 2.3, set-up).
# usage: tools/machine_info.sh > results/machine_<name>.txt
echo "date: $(date -Iseconds)"
echo "== CPU"
lscpu | grep -E "Model name|^CPU\(s\)|Thread\(s\) per core|Core\(s\) per socket|Socket\(s\)|MHz"
echo "== memory"
free -h | head -2
echo "== OS"
uname -srm
grep PRETTY_NAME /etc/os-release
echo "== compiler"
g++ --version | head -1
cmake --version | head -1
echo "== GPU"
if command -v nvidia-smi > /dev/null; then
  nvidia-smi --query-gpu=name,driver_version,clocks.max.sm,memory.total --format=csv
else
  echo "no nvidia-smi"
fi
if command -v nvcc > /dev/null; then nvcc --version | tail -2; else echo "no nvcc"; fi
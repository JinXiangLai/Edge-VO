program=$1
sudo perf record -F 997 -g --call-graph dwarf $program
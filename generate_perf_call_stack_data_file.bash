# 生成含调用栈的.data文件以便画svg图
pid=$1
sample_time=$2 # seconds
sudo perf record -F 997 -p ${pid} -g --call-graph dwarf -- sleep ${sample_time}

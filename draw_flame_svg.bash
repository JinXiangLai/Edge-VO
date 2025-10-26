# 画火焰图
perf_data="$1"
flamegraph_path="$2"
output="$3"

echo "生成火焰图..."
echo "输入: $perf_data"
echo "输出: $output"

# 生成火焰图
perf script -i "$perf_data" | \
"$flamegraph_path/stackcollapse-perf.pl" | \
"$flamegraph_path/flamegraph.pl" > "$output"
#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/tianruiming/CE_ADA_LLAMA"
BUILD_DIR="$ROOT_DIR/build-release-current"
MODEL_HOST_PATH="$ROOT_DIR/src/llama.cpp/gguf_models/qwen.gguf.sort_svd.compact.gguf"
CGROUP_PATH="/sys/fs/cgroup/tianruiming-exclusive"
ANDROID_DIR="/data/local/tmp/CE_Ada"
ANDROID_MODEL="$ANDROID_DIR/qwen.gguf.sort_svd.compact.gguf"
ADB_SERIAL="${ADB_SERIAL:-10.20.0.3:5555}"
PORT="${PORT:-7788}"
TOKENS="${TOKENS:-8}"
SERVER_THREADS="${SERVER_THREADS:-8}"
OFFLOAD_RATE="${OFFLOAD_RATE:-1.0}"
RESULT_ROOT="${RESULT_ROOT:-$ROOT_DIR/src/llama.cpp/exp6_decode_svd_model/results}"
RUN_TAG="${RUN_TAG:-android_phone_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="$RESULT_ROOT/$RUN_TAG"

mkdir -p "$RUN_DIR"

declare -a CLIENT_CONFIGS=(
  "8 68-75"
  "6 68-73"
  "4 68-71"
  "2 68-69"
  "1 68"
)

join_cgroup() {
  echo $$ | sudo tee "$CGROUP_PATH/cgroup.procs" >/dev/null
}

stop_phone_server() {
  adb -s "$ADB_SERIAL" shell "pkill -f '/data/local/tmp/CE_Ada/svd_mobile_server' || true" >/dev/null 2>&1 || true
  adb -s "$ADB_SERIAL" forward --remove "tcp:$PORT" >/dev/null 2>&1 || true
}

start_phone_server() {
  local server_log="$1"
  stop_phone_server
  adb -s "$ADB_SERIAL" forward "tcp:$PORT" "tcp:$PORT"
  adb -s "$ADB_SERIAL" shell "cd $ANDROID_DIR && chmod +x svd_mobile_server && nohup ./svd_mobile_server $ANDROID_MODEL $PORT $SERVER_THREADS > phone_server.log 2>&1 &"
  sleep 5
  adb -s "$ADB_SERIAL" shell "cat $ANDROID_DIR/phone_server.log" >"$server_log" 2>&1 || true
}

run_one() {
  local client_threads="$1"
  local client_cpus="$2"
  local prefix="pc${client_threads}_phone${SERVER_THREADS}_real"
  local server_log="$RUN_DIR/${prefix}_server.log"
  local client_log="$RUN_DIR/${prefix}_client.log"

  echo "== Running $prefix =="
  start_phone_server "$server_log"

  join_cgroup
  (
    cd "$BUILD_DIR"
    join_cgroup
    exec taskset -c "$client_cpus" env LD_LIBRARY_PATH=./bin OMP_NUM_THREADS="$client_threads" \
      ./decode_svd_test "$MODEL_HOST_PATH" "$TOKENS" "$client_threads" 0 "127.0.0.1:$PORT" "$OFFLOAD_RATE"
  ) >"$client_log" 2>&1

  adb -s "$ADB_SERIAL" shell "cat $ANDROID_DIR/phone_server.log" >>"$server_log" 2>&1 || true

  local decode_tps
  decode_tps="$(grep -F "Decode-only throughput:" "$client_log" | tail -n1 | awk '{print $3}')"
  local e2e_tps
  e2e_tps="$(grep -F "End-to-end throughput:" "$client_log" | tail -n1 | awk '{print $3}')"

  printf "%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$prefix" "$client_threads" "$client_cpus" "$SERVER_THREADS" "${decode_tps:-NA}" "${e2e_tps:-NA}" \
    >>"$RUN_DIR/summary.tsv"
}

trap stop_phone_server EXIT

join_cgroup
printf "config\tpc_threads\tpc_cpus\tphone_threads\tdecode_tps\te2e_tps\n" >"$RUN_DIR/summary.tsv"

for entry in "${CLIENT_CONFIGS[@]}"; do
  read -r client_threads client_cpus <<<"$entry"
  run_one "$client_threads" "$client_cpus"
done

echo "Results written to $RUN_DIR"

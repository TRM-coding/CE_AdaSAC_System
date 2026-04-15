#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/tianruiming/CE_ADA_LLAMA"
BUILD_DIR="$ROOT_DIR/build-release-current"
MODEL_PATH="$ROOT_DIR/src/llama.cpp/gguf_models/qwen.gguf.sort_svd.compact.gguf"
SERVER_MODEL_PATH="${SERVER_MODEL_PATH:-$MODEL_PATH}"
SERVER_QUANT_MODE="${SERVER_QUANT_MODE:-off}"
CGROUP_PATH="/sys/fs/cgroup/tianruiming-exclusive"
PORT="${PORT:-7788}"
TOKENS="${TOKENS:-8}"
SERVER_THREADS="${SERVER_THREADS:-8}"
OFFLOAD_RATE="${OFFLOAD_RATE:-1.0}"
RESULT_ROOT="${RESULT_ROOT:-$ROOT_DIR/src/llama.cpp/exp6_decode_svd_model/results}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="$RESULT_ROOT/$RUN_TAG"

mkdir -p "$RUN_DIR"

SERVER_CPUS="60-67"
declare -a CLIENT_CONFIGS=(
  "8 68-75"
  "6 68-73"
  "4 68-71"
  "2 68-69"
)

join_cgroup() {
  echo $$ | sudo tee "$CGROUP_PATH/cgroup.procs" >/dev/null
}

cleanup_server() {
  if [[ -n "${SERVER_PID:-}" ]]; then
    kill "$SERVER_PID" >/dev/null 2>&1 || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}

run_one() {
  local client_threads="$1"
  local client_cpus="$2"
  local prefix="pc${client_threads}_phone${SERVER_THREADS}"
  local server_log="$RUN_DIR/${prefix}_server.log"
  local client_log="$RUN_DIR/${prefix}_client.log"

  echo "== Running $prefix =="
  join_cgroup
  (
    cd "$BUILD_DIR"
    join_cgroup
    exec taskset -c "$SERVER_CPUS" env LD_LIBRARY_PATH=./bin OMP_NUM_THREADS="$SERVER_THREADS" \
      ./svd_mobile_server "$SERVER_MODEL_PATH" "$PORT" "$SERVER_THREADS" "$SERVER_QUANT_MODE"
  ) >"$server_log" 2>&1 &
  SERVER_PID=$!

  sleep 3

  join_cgroup
  (
    cd "$BUILD_DIR"
    join_cgroup
    exec taskset -c "$client_cpus" env LD_LIBRARY_PATH=./bin OMP_NUM_THREADS="$client_threads" \
      ./decode_svd_test "$MODEL_PATH" "$TOKENS" "$client_threads" 0 "127.0.0.1:$PORT" "$OFFLOAD_RATE"
  ) >"$client_log" 2>&1

  cleanup_server
  SERVER_PID=""

  local decode_tps
  decode_tps="$(grep -F "Decode-only throughput:" "$client_log" | tail -n1 | awk '{print $3}')"
  local e2e_tps
  e2e_tps="$(grep -F "End-to-end throughput:" "$client_log" | tail -n1 | awk '{print $3}')"

  {
    printf "%s\t%s\t%s\t%s\t%s\t%s\n" \
      "$prefix" "$client_threads" "$client_cpus" "$SERVER_THREADS" "${decode_tps:-NA}" "${e2e_tps:-NA}"
  } >>"$RUN_DIR/summary.tsv"
}

trap cleanup_server EXIT

join_cgroup
printf "config\tpc_threads\tpc_cpus\tphone_threads\tdecode_tps\te2e_tps\n" >"$RUN_DIR/summary.tsv"

for entry in "${CLIENT_CONFIGS[@]}"; do
  read -r client_threads client_cpus <<<"$entry"
  run_one "$client_threads" "$client_cpus"
done

echo "Results written to $RUN_DIR"

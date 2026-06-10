#!/usr/bin/env bash
set -euo pipefail

APP_DIR="${APP_DIR:-/opt/path-rna-fusion}"
cd "$APP_DIR"

RUNPOD_MODE="${RUNPOD_MODE:-${MODE:-server}}"
DB_PATH="${DB_PATH:-/workspace/server_state/runpod_distributed.sqlite}"
SERVER_HOST="${SERVER_HOST:-0.0.0.0}"
SERVER_PORT="${SERVER_PORT:-8080}"
WORKER_TOKEN="${WORKER_TOKEN:-}"
RUN_ID="${RUN_ID:-default}"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-/workspace}"
LOCAL_WSI_CACHE_DIR="${LOCAL_WSI_CACHE_DIR:-/tmp/wsi_cache}"

login_huggingface() {
  if [[ -n "${HF_TOKEN:-}" ]]; then
    huggingface-cli login --token "$HF_TOKEN" >/tmp/hf_login.log 2>&1 || {
      cat /tmp/hf_login.log >&2
      exit 1
    }
  fi
}

warmup_uni2h() {
  if [[ "${WARMUP_UNI2H:-0}" == "1" ]]; then
    login_huggingface
    python - <<'PY'
import os
from scripts.extract_uni2h_features import DEFAULT_CONFIG, load_model
device = os.environ.get("WARMUP_DEVICE", "cpu")
model = load_model(DEFAULT_CONFIG, device)
print({"uni2h_warmup": True, "model": type(model).__name__, "device": device})
PY
  fi
}

case "$RUNPOD_MODE" in
  server)
    if [[ -z "$WORKER_TOKEN" ]]; then
      echo "WORKER_TOKEN is required for RUNPOD_MODE=server" >&2
      exit 2
    fi
    if [[ "${RESET_DB:-0}" == "1" ]]; then
      python -m scripts.distributed.cli --db "$DB_PATH" init-db --reset
    else
      python -m scripts.distributed.cli --db "$DB_PATH" init-db
    fi
    exec python -m scripts.distributed.cli \
      --db "$DB_PATH" \
      serve \
      --host "$SERVER_HOST" \
      --port "$SERVER_PORT" \
      --token "$WORKER_TOKEN"
    ;;
  wsi-worker)
    if [[ -z "${SERVER_URL:-}" ]]; then
      echo "SERVER_URL is required for RUNPOD_MODE=wsi-worker" >&2
      exit 2
    fi
    if [[ -z "$WORKER_TOKEN" ]]; then
      echo "WORKER_TOKEN is required for RUNPOD_MODE=wsi-worker" >&2
      exit 2
    fi
    login_huggingface
    warmup_uni2h
    exec python -m scripts.distributed.cli wsi-worker \
      --server-url "$SERVER_URL" \
      --token "$WORKER_TOKEN" \
      --run-id "$RUN_ID" \
      --workspace-root "$WORKSPACE_ROOT" \
      --local-cache-dir "$LOCAL_WSI_CACHE_DIR" \
      --batch-jobs "${BATCH_JOBS:-4}" \
      --prefetch-jobs "${PREFETCH_JOBS:-2}" \
      --prefetch-max-bytes "${PREFETCH_MAX_BYTES:-250GB}" \
      --lease-seconds "${LEASE_SECONDS:-1800}"
    ;;
  worker)
    exec python -m scripts.distributed.cli worker
    ;;
  warmup-uni2h)
    login_huggingface
    WARMUP_UNI2H=1 warmup_uni2h
    ;;
  bash)
    exec bash
    ;;
  *)
    exec "$@"
    ;;
esac

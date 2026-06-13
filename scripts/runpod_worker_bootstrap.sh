#!/usr/bin/env bash
set -Eeuo pipefail

APP_DIR="${APP_DIR:-/workspace/uni2h-preprocessing}"
BOOTSTRAP_LOG_DIR="$APP_DIR/logs/runpod-workers"
mkdir -p "$BOOTSTRAP_LOG_DIR"
BOOTSTRAP_LOG="$BOOTSTRAP_LOG_DIR/bootstrap_${HOSTNAME:-worker}_$(date -u +%Y%m%dT%H%M%SZ).log"
touch "$BOOTSTRAP_LOG"
exec >>"$BOOTSTRAP_LOG" 2>&1

on_error() {
  code="$?"
  line="${1:-unknown}"
  echo "[bootstrap] failed code=$code line=$line at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "[bootstrap] log=$BOOTSTRAP_LOG"
  echo "[bootstrap] keeping container alive for SSH debugging"
  tail -f /dev/null
}
trap 'on_error $LINENO' ERR

echo "[bootstrap] start host=${HOSTNAME:-unknown} app_dir=$APP_DIR at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
cd "$APP_DIR"

echo "[bootstrap] repo files:"
ls -lah scripts configs || true

apt-get update
apt-get install -y --no-install-recommends \
  ca-certificates \
  curl \
  git \
  libgl1 \
  libglib2.0-0 \
  libjpeg-turbo8 \
  libopenjp2-7 \
  libopenslide0 \
  openslide-tools
rm -rf /var/lib/apt/lists/*

python -m pip install --upgrade pip
python -m pip install -r scripts/requirements-uni2h.txt -r scripts/requirements-runpod.txt
echo "[bootstrap] python deps installed"

if [ -n "${HF_TOKEN:-}" ]; then
  huggingface-cli login --token "$HF_TOKEN" >/tmp/hf_login.log 2>&1 || {
    cat /tmp/hf_login.log >&2
    exit 1
  }
  echo "[bootstrap] huggingface login completed"
fi

echo "[bootstrap] starting worker loop"
exec python -m scripts.distributed.cli wsi-worker \
  --server-url "$SERVER_URL" \
  --token "$WORKER_TOKEN" \
  --run-id "$RUN_ID" \
  --workspace-root "$WORKSPACE_ROOT" \
  --local-cache-dir "${LOCAL_WSI_CACHE_DIR:-/tmp/wsi_cache}" \
  --batch-jobs "${BATCH_JOBS:-4}" \
  --prefetch-jobs "${PREFETCH_JOBS:-2}" \
  --prefetch-max-bytes "${PREFETCH_MAX_BYTES:-250GB}" \
  --lease-seconds "${LEASE_SECONDS:-1800}"

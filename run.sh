#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

FPS="${FPS:-3}"
KEEP_PER_VIDEO="${KEEP_PER_VIDEO:-30}"
TOP_PERCENT="${TOP_PERCENT:-30}"
MIN_SCORE="${MIN_SCORE:-2.0}"
DEDUP_DELTA="${DEDUP_DELTA:-4.0}"
CATEGORIES="${CATEGORIES:-male female}"
LIMIT_VIDEOS="${LIMIT_VIDEOS:-}"

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "未找到 ffmpeg，请先安装并确保在 PATH 中。" >&2
  exit 2
fi
if ! command -v ffprobe >/dev/null 2>&1; then
  echo "未找到 ffprobe，请先安装并确保在 PATH 中。" >&2
  exit 2
fi

if [[ ! -d ".venv" ]]; then
  python3 -m venv .venv
fi

source .venv/bin/activate
python -m pip install -r requirements.txt >/dev/null

CMD=(python extract_clear_frames.py
  --fps "$FPS"
  --keep-per-video "$KEEP_PER_VIDEO"
  --top-percent "$TOP_PERCENT"
  --min-score "$MIN_SCORE"
  --dedup
  --dedup-delta "$DEDUP_DELTA"
  --categories
)

IFS=' ' read -r -a CAT_ARR <<< "$CATEGORIES"
CMD+=("${CAT_ARR[@]}")

if [[ -n "$LIMIT_VIDEOS" ]]; then
  CMD+=(--limit-videos "$LIMIT_VIDEOS")
fi

echo "运行参数：fps=$FPS keep_per_video=$KEEP_PER_VIDEO top_percent=$TOP_PERCENT min_score=$MIN_SCORE dedup_delta=$DEDUP_DELTA categories=($CATEGORIES) limit_videos=${LIMIT_VIDEOS:-all}"
exec "${CMD[@]}"


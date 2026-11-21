#!/bin/bash
# bash scripts/helipad_score_pod.sh <best.pt> <out.json> [limit_cases, 0 = all] -- kept for the old call; the launcher is helipad_score_pod.py
exec uv run --no-project python "$(dirname "$0")/helipad_score_pod.py" "$1" "$2" --limit-cases "${3:-0}"

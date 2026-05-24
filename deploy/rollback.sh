#!/usr/bin/env bash
# Copyright (c) 2026 <Your Name(s)>
# Tatung University — I4210 AI實務專題
# deploy/rollback.sh — revert to the previously-deployed tag in <30 s
set -euo pipefail

STATE_DIR=/var/lib/edgeai-hw6

if [ ! -f "$STATE_DIR/deployed.txt" ]; then
  echo "[rollback] ERROR: no deployed.txt — nothing to roll back from" >&2
  exit 1
fi

CURRENT=$(cat "$STATE_DIR/deployed.txt")

if [ ! -f "$STATE_DIR/deployed.txt.history" ]; then
  echo "[rollback] ERROR: no history file — cannot determine previous tag" >&2
  exit 1
fi

PREV=$(tail -1 "$STATE_DIR/deployed.txt.history")
if [ -z "$PREV" ]; then
  echo "[rollback] ERROR: history file is empty" >&2
  exit 1
fi

echo "[rollback] Rolling back: $CURRENT → $PREV"

# Switch to previous image and restart (no pull — must be in local cache).
export IMAGE_TAG="$PREV"
docker compose -f deploy/docker-compose.yml up -d --force-recreate

if ! bash deploy/healthcheck.sh; then
  echo "[rollback] Healthcheck failed after rollback — manual intervention required" >&2
  exit 1
fi

# Update state: remove last history line, write prev as current.
head -n -1 "$STATE_DIR/deployed.txt.history" > "$STATE_DIR/deployed.txt.history.tmp"
mv "$STATE_DIR/deployed.txt.history.tmp" "$STATE_DIR/deployed.txt.history"
echo "$PREV" > "$STATE_DIR/deployed.txt"

echo "[rollback] Successfully rolled back to $PREV"

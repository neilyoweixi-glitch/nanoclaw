#!/bin/bash
# Autonomous ML Research Runner v2
# Simplified version with proper output capture

set -e
cd /Volumes/codespace/nanoclaw/groups/autoresearch

# Config
SLACK_CHANNEL="C0AL2B76538"
SLACK_TOKEN=$(grep SLACK_BOT_TOKEN /Volumes/codespace/nanoclaw/.env | cut -d= -f2)

# Ensure venv
source .venv/bin/activate

LOG_FILE="research_$(date +%Y%m%d_%H%M%S).log"

send_slack() {
    local msg="$1"
    curl -s -X POST "https://slack.com/api/chat.postMessage" \
        -H "Authorization: Bearer $SLACK_TOKEN" \
        -H "Content-Type: application/json" \
        -d "{\"channel\":\"$SLACK_CHANNEL\",\"text\":\"$msg\"}" > /dev/null
}

log() {
    echo "[$(date '+%H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log "=== AutoResearch Started ==="
send_slack "🚀 AutoResearch: Starting autonomous experiments on Mac Mini MPS"

EXPERIMENT=0
while true; do
    EXPERIMENT=$((EXPERIMENT + 1))
    START=$(date +%s)

    log "--- Experiment $EXPERIMENT ---"

    # Run training
    log "Running training..."
    python -u train.py 2>&1 | tee -a "$LOG_FILE"
    TRAIN_EXIT=$?

    END=$(date +%s)
    DURATION=$((END - START))

    # Extract final val_bpb (format: "val_bpb:          12.004855")
    VAL_BPB=$(grep "^val_bpb:" "$LOG_FILE" | tail -1 | awk '{print $2}')

    log "Experiment $EXPERIMENT done: val_bpb=$VAL_BPB, duration=${DURATION}s"

    # Report to Slack
    send_slack "🔬 Exp #$EXPERIMENT: val_bpb=$VAL_BPB (${DURATION}s)"

    # Brief pause
    sleep 10
done

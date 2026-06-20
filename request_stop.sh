#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
touch "$DIR/runtime_stop.flag"
echo "BungVision out-of-band stop requested: $DIR/runtime_stop.flag"
if [ -f "$DIR/runtime_stop_ack.txt" ]; then
  echo "Previous ack:"
  tail -n 1 "$DIR/runtime_stop_ack.txt" || true
fi

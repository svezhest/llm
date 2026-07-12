#!/usr/bin/env bash
# Liveness watchdog: if the gateway /health stops answering, kick the service.
# Schedule via cron or a StartInterval launchd agent (every 30s).
set -uo pipefail
URL="${GATEWAY_HEALTH:-http://127.0.0.1:8080/health}"
if ! curl -fsS --max-time 3 "$URL" >/dev/null 2>&1; then
  echo "$(date) gateway unhealthy -> kickstart"
  launchctl kickstart -k "gui/$(id -u)/com.qwen.gateway"
fi

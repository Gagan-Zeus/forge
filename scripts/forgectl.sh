#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME_DIR="$ROOT_DIR/.forge"
PID_FILE="$RUNTIME_DIR/forge.pid"
LOG_FILE="$RUNTIME_DIR/forge.log"
RUN_PATTERN="[Pp]ython(.*/python)? .*run\.py"

resolve_python() {
    if [[ -n "${PYTHON_BIN:-}" ]]; then
        if [[ ! -x "$PYTHON_BIN" ]]; then
            echo "PYTHON_BIN is set but not executable: $PYTHON_BIN" >&2
            exit 1
        fi
        return
    fi

    if [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
        PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
        return
    fi

    if command -v python3 >/dev/null 2>&1; then
        PYTHON_BIN="$(command -v python3)"
        return
    fi

    if command -v python >/dev/null 2>&1; then
        PYTHON_BIN="$(command -v python)"
        return
    fi

    echo "No Python executable found. Set PYTHON_BIN or create .venv." >&2
    exit 1
}

read_pid() {
    if [[ ! -f "$PID_FILE" ]]; then
        return 1
    fi

    local pid
    pid="$(cat "$PID_FILE" 2>/dev/null || true)"
    if [[ "$pid" =~ ^[0-9]+$ ]]; then
        echo "$pid"
        return 0
    fi

    return 1
}

is_running() {
    local pid
    pid="$(read_pid)" || return 1
    kill -0 "$pid" 2>/dev/null
}

list_runpy_pids() {
    if ! command -v pgrep >/dev/null 2>&1; then
        return 0
    fi

    local out
    out="$(pgrep -f "$RUN_PATTERN" 2>/dev/null || true)"
    if [[ -n "$out" ]]; then
        printf '%s\n' "$out"
    fi
}

list_all_forge_pids() {
    local printed_any=0

    if is_running; then
        read_pid
        printed_any=1
    fi

    local pid
    while IFS= read -r pid; do
        [[ -z "$pid" ]] && continue
        if [[ "$printed_any" -eq 1 && "$pid" == "$(read_pid 2>/dev/null || true)" ]]; then
            continue
        fi
        echo "$pid"
        printed_any=1
    done < <(list_runpy_pids)
}

wait_for_exit() {
    local pid="$1"
    local attempts=25

    while (( attempts > 0 )); do
        if ! kill -0 "$pid" 2>/dev/null; then
            return 0
        fi
        attempts=$((attempts - 1))
        sleep 0.2
    done

    return 1
}

start() {
    local existing_pid
    existing_pid="$(list_all_forge_pids | head -n 1 || true)"
    if [[ -n "$existing_pid" ]]; then
        echo "Forge is already running (PID $existing_pid)."
        echo "$existing_pid" >"$PID_FILE"
        exit 0
    fi

    mkdir -p "$RUNTIME_DIR"
    resolve_python

    (
        cd "$ROOT_DIR"
        nohup "$PYTHON_BIN" run.py >>"$LOG_FILE" 2>&1 &
        echo "$!" >"$PID_FILE"
    )

    if is_running; then
        echo "Forge started in background (PID $(read_pid))."
        echo "Logs: $LOG_FILE"
        exit 0
    fi

    echo "Forge failed to start. Check logs: $LOG_FILE" >&2
    rm -f "$PID_FILE"
    exit 1
}

stop() {
    local pids
    pids="$(list_all_forge_pids | awk '!seen[$0]++' || true)"
    if [[ -z "$pids" ]]; then
        echo "Forge is not running."
        rm -f "$PID_FILE"
        exit 0
    fi

    local pid
    while IFS= read -r pid; do
        [[ -z "$pid" ]] && continue
        kill "$pid" 2>/dev/null || true
    done <<<"$pids"

    local stubborn
    stubborn=""
    while IFS= read -r pid; do
        [[ -z "$pid" ]] && continue
        if ! wait_for_exit "$pid"; then
            stubborn+="$pid "$'
'
        fi
    done <<<"$pids"

    if [[ -n "$stubborn" ]]; then
        while IFS= read -r pid; do
            [[ -z "$pid" ]] && continue
            kill -9 "$pid" 2>/dev/null || true
        done <<<"$stubborn"
    fi

    rm -f "$PID_FILE"
    echo "Forge stop completed for PID(s):"
    echo "$pids" | sed '/^$/d' | sed 's/^/- /'
}

status() {
    local pids
    pids="$(list_all_forge_pids | awk '!seen[$0]++' || true)"
    if [[ -n "$pids" ]]; then
        local primary
        primary="$(echo "$pids" | head -n 1)"
        echo "Forge is running (PID $primary)."
        echo "$primary" >"$PID_FILE"
        echo "Logs: $LOG_FILE"
        if [[ "$(echo "$pids" | wc -l | tr -d ' ')" -gt 1 ]]; then
            echo "Additional PID(s):"
            echo "$pids" | tail -n +2 | sed '/^$/d' | sed 's/^/- /'
        fi
        exit 0
    fi

    echo "Forge is not running."
    if [[ -f "$PID_FILE" ]]; then
        echo "Removing stale PID file: $PID_FILE"
        rm -f "$PID_FILE"
    fi
}

logs() {
    if [[ ! -f "$LOG_FILE" ]]; then
        echo "No logs yet: $LOG_FILE"
        exit 0
    fi
    tail -n 120 -f "$LOG_FILE"
}

usage() {
    cat <<'EOF'
Usage: ./scripts/forgectl.sh <command>

Commands:
  start   Start Forge in background
  stop    Stop background Forge process
  status  Show process status
  logs    Stream recent logs
EOF
}

command="${1:-}"
case "$command" in
    start)
        start
        ;;
    stop)
        stop
        ;;
    status)
        status
        ;;
    logs)
        logs
        ;;
    *)
        usage
        exit 1
        ;;
esac

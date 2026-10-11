#!/usr/bin/env bash
# =============================================================================
# HumanBot one-click launcher (Linux / macOS)
#
#   ./start.sh                 normal start
#   ./start.sh --port 8899     force a port
#   ./start.sh --no-bots       do not log in bots
#   ./start.sh --reset-password regenerate console password
#
# Everything below is plain ASCII on purpose, so the file runs correctly
# regardless of the system locale or editor encoding.
# =============================================================================

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

VENV_DIR="$SCRIPT_DIR/.venv"
PY_BIN=""
PORT_FILE="$SCRIPT_DIR/data/.console_port"

say() { printf '%s\n' "$*"; }
warn() { printf '[WARN] %s\n' "$*"; }
die() { printf '[ERROR] %s\n' "$*" >&2; exit 1; }

say "=============================================================="
say "  HumanBot - starting up"
say "=============================================================="

# -----------------------------------------------------------------------------
# 1. Locate a usable Python interpreter
# -----------------------------------------------------------------------------
find_python() {
  local c
  for c in python3 python3.13 python3.12 python3.11 python3.10 python3.9 python; do
    if command -v "$c" >/dev/null 2>&1; then
      if "$c" -c 'import sys; sys.exit(0 if sys.version_info >= (3,9) else 1)' 2>/dev/null; then
        PY_BIN="$(command -v "$c")"
        return 0
      fi
    fi
  done
  return 1
}

if ! find_python; then
  die "Python 3.9+ not found. Please install Python first."
fi
say "[1/6] Python found: $PY_BIN ($("$PY_BIN" -c 'import platform;print(platform.python_version())'))"

# -----------------------------------------------------------------------------
# 2. Create / reuse virtualenv
# -----------------------------------------------------------------------------
if [ ! -d "$VENV_DIR" ]; then
  say "[2/6] Creating virtual environment (.venv) ..."
  "$PY_BIN" -m venv "$VENV_DIR" || die "Failed to create virtualenv."
else
  say "[2/6] Reusing virtual environment (.venv)"
fi

if [ -x "$VENV_DIR/bin/python" ]; then
  VP="$VENV_DIR/bin/python"
elif [ -x "$VENV_DIR/Scripts/python.exe" ]; then
  VP="$VENV_DIR/Scripts/python.exe"
else
  die "virtualenv looks broken. Delete .venv and run again."
fi

# -----------------------------------------------------------------------------
# 3. Install dependencies
# -----------------------------------------------------------------------------
STAMP="$VENV_DIR/.deps_ok"
NEED_INSTALL=0
if [ ! -f "$STAMP" ]; then
  NEED_INSTALL=1
elif [ "$SCRIPT_DIR/requirements.txt" -nt "$STAMP" ]; then
  NEED_INSTALL=1
fi

if [ "$NEED_INSTALL" = "1" ]; then
  say "[3/6] Installing dependencies (first run may take a few minutes) ..."
  "$VP" -m pip install --upgrade pip -q 2>/dev/null || true
  if "$VP" -m pip install -r "$SCRIPT_DIR/requirements.txt" -q; then
    touch "$STAMP"
    say "      dependencies installed."
  else
    warn "quiet install failed, retrying with full output ..."
    "$VP" -m pip install -r "$SCRIPT_DIR/requirements.txt" || die "Dependency install failed."
    touch "$STAMP"
  fi
else
  say "[3/6] Dependencies already installed (skip)"
fi

# -----------------------------------------------------------------------------
# 4. Make sure runtime folders exist
# -----------------------------------------------------------------------------
if [ ! -f "$SCRIPT_DIR/config.yaml" ]; then
  die "config.yaml not found. Please restore it before starting."
fi
if [ ! -f "$SCRIPT_DIR/persona.yaml" ]; then
  warn "persona.yaml not found, the built-in default persona will be used."
fi
mkdir -p "$SCRIPT_DIR/data" "$SCRIPT_DIR/logs" "$SCRIPT_DIR/assets" 2>/dev/null

say "[4/6] Runtime folders ready (data / logs / assets)"

# -----------------------------------------------------------------------------
# 5. Free-port detection (mirrors the logic inside the app)
# -----------------------------------------------------------------------------
pick_port() {
  "$VP" - <<'PYEOF'
import socket
for port in range(8800, 8901):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        s.bind(("0.0.0.0", port))
        s.close()
        print(port)
        raise SystemExit
    except OSError:
        s.close()
        continue
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind(("0.0.0.0", 0))
print(s.getsockname()[1])
s.close()
PYEOF
}

# Only auto-pick when the user did not pass --port
REQUESTED_PORT=""
for a in "$@"; do
  case "$a" in
    --port=*) REQUESTED_PORT="${a#--port=}" ;;
    --port)   shift 0 ;;
  esac
done
args=("$@")
i=0
while [ $i -lt $# ]; do
  if [ "${args[$i]}" = "--port" ]; then
    j=$((i + 1))
    if [ $j -lt $# ]; then REQUESTED_PORT="${args[$j]}"; fi
  fi
  i=$((i + 1))
done

if [ -n "$REQUESTED_PORT" ]; then
  FREE_PORT="$REQUESTED_PORT"
  say "[5/6] Using requested port: $FREE_PORT"
else
  FREE_PORT="$(pick_port)"
  if [ -z "$FREE_PORT" ]; then
    die "Could not find a free port."
  fi
  say "[5/6] Auto-selected free port: $FREE_PORT"
fi
printf '%s' "$FREE_PORT" > "$PORT_FILE" 2>/dev/null

# -----------------------------------------------------------------------------
# 6. Launch
# -----------------------------------------------------------------------------
say "[6/6] Launching HumanBot ..."
say ""

if [ -n "$REQUESTED_PORT" ]; then
  exec "$VP" -m humanbot.main --port "$REQUESTED_PORT" "$@"
else
  # strip nothing: the app performs the same free-port scan when port is omitted
  exec "$VP" -m humanbot.main "$@"
fi

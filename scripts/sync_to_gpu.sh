#!/usr/bin/env bash
set -euo pipefail

HOST="gpu4060"
REMOTE_DIR="/home/ubuntu/Developer/long-horizon-dynamics"
WITH_DATA=0
DELETE=0
DRY_RUN=0

usage() {
  cat <<'USAGE'
Usage: scripts/sync_to_gpu.sh [options]

Sync this repository from the local machine to a remote GPU host.

Options:
  --host HOST           Remote SSH host (default: gpu4060)
  --remote-dir DIR      Remote destination directory
                        (default: /home/ubuntu/Developer/long-horizon-dynamics)
  --with-data           Include large local datasets normally excluded
  --delete              Delete remote files that are absent locally
  --dry-run             Show what would be synced without changing remote files
  -h, --help            Show this help
USAGE
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --host)
      [ "$#" -ge 2 ] || { echo "error: --host requires a value" >&2; exit 2; }
      HOST="$2"
      shift 2
      ;;
    --remote-dir)
      [ "$#" -ge 2 ] || { echo "error: --remote-dir requires a value" >&2; exit 2; }
      REMOTE_DIR="$2"
      shift 2
      ;;
    --with-data)
      WITH_DATA=1
      shift
      ;;
    --delete)
      DELETE=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

RSYNC_ARGS=(
  -a
  --human-readable
  --stats
  --exclude='__pycache__/'
  --exclude='*.py[cod]'
  --exclude='.pytest_cache/'
  --exclude='.mypy_cache/'
  --exclude='.ruff_cache/'
  --exclude='.ipynb_checkpoints/'
  --exclude='.venv/'
  --exclude='venv/'
  --exclude='env/'
  --exclude='.env/'
  --exclude='.conda/'
  --exclude='logs/'
  --exclude='log/'
  --exclude='*.log'
  --exclude='resources/experiments/'
  --exclude='.DS_Store'
)

if [ "$WITH_DATA" -eq 0 ]; then
  RSYNC_ARGS+=(-z)
  RSYNC_ARGS+=(
    --exclude='resources/data/pi_tcn/'
    --exclude='resources/data/neurobem/'
    --exclude='resources/data.zip'
  )
else
  RSYNC_ARGS+=(--partial --whole-file)
fi

if [ "$DELETE" -eq 1 ]; then
  RSYNC_ARGS+=(--delete)
fi

if [ "$DRY_RUN" -eq 1 ]; then
  RSYNC_ARGS+=(--dry-run --itemize-changes)
fi

echo "Syncing $REPO_ROOT/ to $HOST:$REMOTE_DIR/"
[ "$WITH_DATA" -eq 1 ] || echo "Large data excluded; pass --with-data to include it."

ssh "$HOST" "mkdir -p '$REMOTE_DIR'"
rsync "${RSYNC_ARGS[@]}" "$REPO_ROOT/" "$HOST:$REMOTE_DIR/"

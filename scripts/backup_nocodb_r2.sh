#!/usr/bin/env bash
set -euo pipefail

CONFIG_FILE="${CONFIG_FILE:-$HOME/.config/jarvis/nocodb-r2-backup.env}"

if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "Missing config file: $CONFIG_FILE" >&2
  exit 2
fi

# shellcheck source=/dev/null
source "$CONFIG_FILE"

: "${NOCODB_DB_PATH:?Set NOCODB_DB_PATH}"
: "${BACKUP_WORK_DIR:?Set BACKUP_WORK_DIR}"
: "${RCLONE_REMOTE:?Set RCLONE_REMOTE}"
: "${RCLONE_DEST:?Set RCLONE_DEST}"
: "${GPG_PASSPHRASE_FILE:?Set GPG_PASSPHRASE_FILE}"

NOCODB_DATA_DIR="${NOCODB_DATA_DIR:-$(dirname "$NOCODB_DB_PATH")}"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
SNAPSHOT_DIR="$BACKUP_WORK_DIR/snapshots"
ARCHIVE_DIR="$BACKUP_WORK_DIR/archives"
TMP_DIR="$BACKUP_WORK_DIR/tmp-$TIMESTAMP"

mkdir -p "$SNAPSHOT_DIR" "$ARCHIVE_DIR" "$TMP_DIR"
chmod 700 "$BACKUP_WORK_DIR" "$SNAPSHOT_DIR" "$ARCHIVE_DIR" "$TMP_DIR"

SNAPSHOT_DB="$SNAPSHOT_DIR/noco-$TIMESTAMP.db"
ARCHIVE="$ARCHIVE_DIR/nocodb-$TIMESTAMP.tar.gz"
ENCRYPTED="$ARCHIVE.gpg"

cleanup() {
  rm -rf "$TMP_DIR"
}
trap cleanup EXIT

echo "[$(date -Is)] Creating SQLite snapshot: $SNAPSHOT_DB"
python3 - "$NOCODB_DB_PATH" "$SNAPSHOT_DB" <<'PY'
import sqlite3
import sys

source_path, dest_path = sys.argv[1], sys.argv[2]
source = sqlite3.connect(f"file:{source_path}?mode=ro", uri=True)
try:
    dest = sqlite3.connect(dest_path)
    try:
        source.backup(dest)
    finally:
        dest.close()
finally:
    source.close()
PY

echo "[$(date -Is)] Verifying snapshot integrity"
python3 - "$SNAPSHOT_DB" <<'PY'
import sqlite3
import sys

db_path = sys.argv[1]
conn = sqlite3.connect(db_path)
try:
    result = conn.execute("PRAGMA integrity_check").fetchone()
finally:
    conn.close()

if not result or result[0] != "ok":
    raise SystemExit(f"SQLite integrity check failed: {result}")
PY

mkdir -p "$TMP_DIR/nocodb"
cp "$SNAPSHOT_DB" "$TMP_DIR/nocodb/noco.db"
if [[ -d "$NOCODB_DATA_DIR/nc" ]]; then
  cp -a "$NOCODB_DATA_DIR/nc" "$TMP_DIR/nocodb/nc"
fi

echo "[$(date -Is)] Building archive: $ARCHIVE"
tar -C "$TMP_DIR" -czf "$ARCHIVE" nocodb

echo "[$(date -Is)] Encrypting archive: $ENCRYPTED"
gpg --batch --yes --symmetric --cipher-algo AES256 \
  --passphrase-file "$GPG_PASSPHRASE_FILE" \
  --output "$ENCRYPTED" "$ARCHIVE"

echo "[$(date -Is)] Uploading to R2: $RCLONE_REMOTE:$RCLONE_DEST"
rclone copy "$ENCRYPTED" "$RCLONE_REMOTE:$RCLONE_DEST"

echo "[$(date -Is)] Pruning local backup files"
find "$SNAPSHOT_DIR" -type f -name 'noco-*.db' -mtime +"${LOCAL_KEEP_DAYS:-7}" -delete
find "$ARCHIVE_DIR" -type f -name 'nocodb-*.tar.gz' -mtime +"${LOCAL_KEEP_DAYS:-7}" -delete
find "$ARCHIVE_DIR" -type f -name 'nocodb-*.tar.gz.gpg' -mtime +"${LOCAL_KEEP_DAYS:-30}" -delete

if [[ "${REMOTE_PRUNE:-0}" == "1" ]]; then
  echo "[$(date -Is)] Pruning remote encrypted archives older than ${REMOTE_KEEP_DAYS:-90} days"
  rclone delete "$RCLONE_REMOTE:$RCLONE_DEST" \
    --min-age "${REMOTE_KEEP_DAYS:-90}d" \
    --include 'nocodb-*.tar.gz.gpg'
fi

echo "[$(date -Is)] Backup completed: $(basename "$ENCRYPTED")"

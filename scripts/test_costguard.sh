#!/bin/zsh
set -eu

SCRIPT_DIR=${0:A:h}
TMP_DIR=$(mktemp -d)
trap 'rm -rf "$TMP_DIR"' EXIT
mkdir -p "$TMP_DIR/bin" "$TMP_DIR/context"

cat > "$TMP_DIR/bin/runpodctl" <<'EOF'
#!/bin/sh
if [ "$1 $2" = "pod get" ]; then
  printf '%s\n' '{"desiredStatus":"RUNNING","volumeInGb":0,"ssh":{"ip":"192.0.2.1","port":22}}'
else
  printf '%s\n' stop >> "$COSTGUARD_TEST_CALLS"
fi
EOF
cat > "$TMP_DIR/bin/ssh" <<'EOF'
#!/bin/sh
case "$*" in
  *"sync") exit 0 ;;
  *) printf '%s\n' 'METRICS|0|0|0|1' ;;
esac
EOF
cat > "$TMP_DIR/bin/pgrep" <<'EOF'
#!/bin/sh
exit 1
EOF
chmod +x "$TMP_DIR/bin/"*

export PATH="$TMP_DIR/bin:$PATH"
export COSTGUARD_POD_ID=test-pod
export COSTGUARD_CONTEXT_DIR="$TMP_DIR/context"
export COSTGUARD_LOG_FILE="$TMP_DIR/costguard.log"
export COSTGUARD_TEST_CALLS="$TMP_DIR/calls"
export COSTGUARD_ONCE=1

"$SCRIPT_DIR/costguard.sh" >/dev/null
[[ ! -e "$COSTGUARD_TEST_CALLS" ]]
grep -q 'reason=ephemeral_without_archive_receipt' "$COSTGUARD_LOG_FILE"

touch "$COSTGUARD_CONTEXT_DIR/costguard.safe-to-stop.$COSTGUARD_POD_ID"
"$SCRIPT_DIR/costguard.sh" >/dev/null
[[ $(wc -l < "$COSTGUARD_TEST_CALLS") -eq 1 ]]
grep -q 'interval=1800s' "$COSTGUARD_LOG_FILE"

print 'costguard safety check passed'

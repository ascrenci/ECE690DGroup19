#!/usr/bin/env bash
set -euo pipefail

WORKFLOW="build.yml"
ARTIFACT="HelloWorld-ipa"
OUTPUT_DIR="./dist"

echo "==> Starting GitHub Actions workflow..."

START_TIME=$(date +%s)

gh auth switch --user ascrenci
gh workflow run "$WORKFLOW"

echo "==> Waiting for workflow run to appear..."

RUN_ID=""

for _ in {1..30}; do
RUN_ID=$(gh run list \
--workflow="$WORKFLOW" \
--limit=10 \
--json databaseId,createdAt,status \
--jq ".[] | select(.createdAt != null) | select((.createdAt | fromdateiso8601) >= $START_TIME) | .databaseId" | head -n 1)

if [[ -n "$RUN_ID" ]]; then
    break
fi

sleep 2


done

if [[ -z "$RUN_ID" ]]; then
echo "ERROR: Could not find the newly started workflow run."
exit 1
fi

echo "==> Watching workflow run $RUN_ID..."

gh run watch "$RUN_ID" --exit-status

echo "==> Build succeeded."

rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

echo "==> Downloading IPA artifact..."

gh run download "$RUN_ID" \
--name "$ARTIFACT" \
--dir "$OUTPUT_DIR"

IPA=$(find "$OUTPUT_DIR" -type f -name "*.ipa" -print -quit)

if [[ -z "$IPA" ]]; then
echo "ERROR: No .ipa file was found in the downloaded artifact."
exit 1
fi

echo
echo "Build complete:"
echo "$IPA"
#!/usr/bin/env bash
set -euo pipefail

Configuration

WORKFLOW="build.yml"
ARTIFACT="HelloWorld-ipa"
OUTPUT_DIR="./dist"

echo "==> Starting GitHub Actions workflow..."

# Record the current time so we don't accidentally download an older run.

START_TIME=$(date +%s)

gh workflow run "$WORKFLOW"

echo "==> Workflow dispatched. Waiting for GitHub to create the run..."

# GitHub can take a few seconds to register the workflow run.

RUN_ID=""

for _ in {1..30}; do
RUN_ID=$(gh run list
--workflow="$WORKFLOW"
--limit=10
--json databaseId,createdAt,status
--jq ".[] | select(.createdAt != null) | select((.createdAt | fromdateiso8601) >= $START_TIME) | .databaseId"
| head -n 1)

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

# Remove the previous download.

rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

echo "==> Downloading IPA artifact..."

gh run download "$RUN_ID"
--name "$ARTIFACT"
--dir "$OUTPUT_DIR"

IPA=$(find "$OUTPUT_DIR" -type f -name "*.ipa" -print -quit)

if [[ -z "$IPA" ]]; then
echo "ERROR: Workflow succeeded, but no .ipa was found."
exit 1
fi

echo
echo "========================================"
echo " Build complete!"
echo "========================================"
echo
echo "IPA:"
echo " $IPA"
echo
echo "Size:"
du -h "$IPA" | cut -f1
echo
echo "Ready for sideloading."
#!/usr/bin/env bash
# Downloads the IPA artifact from the latest successful build run to ./dist (no new build is triggered)
set -euo pipefail

WORKFLOW="build.yml"
ARTIFACT="HelloWorld-ipa"
OUTPUT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/dist"

if ! command -v gh >/dev/null 2>&1; then
    echo "ERROR: GitHub CLI (gh) not found. Install it from https://cli.github.com and run 'gh auth login'." >&2
    exit 1
fi

echo "==> Finding latest successful run..."
RUN_INFO=$(gh run list \
    --workflow="$WORKFLOW" \
    --status=success \
    --limit=1 \
    --json databaseId,createdAt,headSha \
    --jq '.[0] | select(. != null) | "\(.databaseId) \(.headSha[0:7]) \(.createdAt)"')

if [[ -z "$RUN_INFO" ]]; then
    echo "ERROR: No successful workflow runs found." >&2
    exit 1
fi

read -r RUN_ID SHA CREATED <<< "$RUN_INFO"
echo "==> Latest successful run: $RUN_ID (commit $SHA, $CREATED)"

rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

echo "==> Downloading IPA artifact..."
gh run download "$RUN_ID" --name "$ARTIFACT" --dir "$OUTPUT_DIR" \
    || { echo "ERROR: Artifact download failed (it may have expired)." >&2; exit 1; }

IPA=$(find "$OUTPUT_DIR" -type f -name "*.ipa" -print -quit)

if [[ -z "$IPA" ]]; then
    echo "ERROR: No .ipa file was found in the downloaded artifact." >&2
    exit 1
fi

echo
echo "Download complete:"
echo "$IPA"

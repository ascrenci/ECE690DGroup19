#Requires -Version 5.1
# Triggers the GitHub Actions build, waits for it, and downloads the IPA to ./dist
$ErrorActionPreference = 'Stop'

$Workflow  = 'build.yml'
$Artifact  = 'HelloWorld-ipa'
$OutputDir = Join-Path (Split-Path -Parent $PSScriptRoot) 'dist'

if (-not (Get-Command gh -ErrorAction SilentlyContinue)) {
    throw "GitHub CLI (gh) not found. Install it from https://cli.github.com and run 'gh auth login'."
}

Write-Host '==> Starting GitHub Actions workflow...'
$startTime = [DateTime]::UtcNow.AddSeconds(-5)   # small buffer for clock skew

gh workflow run $Workflow
if ($LASTEXITCODE -ne 0) { throw 'Failed to start workflow.' }

Write-Host '==> Waiting for workflow run to appear...'
$runId = $null
for ($i = 0; $i -lt 30 -and -not $runId; $i++) {
    $runs = gh run list --workflow=$Workflow --limit=10 --json databaseId,createdAt | ConvertFrom-Json
    foreach ($r in $runs) {
        if (([DateTime]$r.createdAt).ToUniversalTime() -ge $startTime) {
            $runId = $r.databaseId
            break
        }
    }
    if (-not $runId) { Start-Sleep -Seconds 2 }
}

if (-not $runId) { throw 'Could not find the newly started workflow run.' }

Write-Host "==> Watching workflow run $runId..."
gh run watch $runId --exit-status
if ($LASTEXITCODE -ne 0) { throw 'Build failed.' }

Write-Host '==> Build succeeded.'

if (Test-Path $OutputDir) { Remove-Item $OutputDir -Recurse -Force }
New-Item -ItemType Directory -Path $OutputDir | Out-Null

Write-Host '==> Downloading IPA artifact...'
gh run download $runId --name $Artifact --dir $OutputDir
if ($LASTEXITCODE -ne 0) { throw 'Artifact download failed.' }

$ipa = Get-ChildItem -Path $OutputDir -Recurse -Filter *.ipa | Select-Object -First 1
if (-not $ipa) { throw 'No .ipa file was found in the downloaded artifact.' }

Write-Host ''
Write-Host 'Build complete:'
Write-Host $ipa.FullName

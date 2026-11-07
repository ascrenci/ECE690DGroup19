#Requires -Version 5.1
# Downloads the IPA artifact from the latest successful build run to ./dist (no new build is triggered)
$ErrorActionPreference = 'Stop'

$Workflow  = 'build.yml'
$Artifact  = 'HelloWorld-ipa'
$OutputDir = Join-Path (Split-Path -Parent $PSScriptRoot) 'dist'

if (-not (Get-Command gh -ErrorAction SilentlyContinue)) {
    throw "GitHub CLI (gh) not found. Install it from https://cli.github.com and run 'gh auth login'."
}

Write-Host '==> Finding latest successful run...'
$runs = gh run list --workflow=$Workflow --status=success --limit=1 --json databaseId,createdAt,headSha | ConvertFrom-Json
if (-not $runs -or @($runs).Count -eq 0) { throw 'No successful workflow runs found.' }

$run = @($runs)[0]
Write-Host "==> Latest successful run: $($run.databaseId) (commit $($run.headSha.Substring(0,7)), $($run.createdAt))"

if (Test-Path $OutputDir) { Remove-Item $OutputDir -Recurse -Force }
New-Item -ItemType Directory -Path $OutputDir | Out-Null

Write-Host '==> Downloading IPA artifact...'
gh run download $run.databaseId --name $Artifact --dir $OutputDir
if ($LASTEXITCODE -ne 0) { throw 'Artifact download failed (it may have expired).' }

$ipa = Get-ChildItem -Path $OutputDir -Recurse -Filter *.ipa | Select-Object -First 1
if (-not $ipa) { throw 'No .ipa file was found in the downloaded artifact.' }

Write-Host ''
Write-Host 'Download complete:'
Write-Host $ipa.FullName

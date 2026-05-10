param(
    [Parameter(Mandatory = $true)]
    [string]$SourcePkl,
    [Parameter(Mandatory = $true)]
    [string]$DestinationName
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$modelsDir = Join-Path $root "models"

if (!(Test-Path -LiteralPath $SourcePkl)) {
    throw "Source file not found: $SourcePkl"
}

if (!(Test-Path -LiteralPath $modelsDir)) {
    New-Item -ItemType Directory -Path $modelsDir | Out-Null
}

$dest = Join-Path $modelsDir $DestinationName
Copy-Item -LiteralPath $SourcePkl -Destination $dest -Force

$srcHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $SourcePkl).Hash
$dstHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $dest).Hash

Write-Host "Promoted to: $dest"
Write-Host "Source SHA256: $srcHash"
Write-Host "Dest   SHA256: $dstHash"

if ($srcHash -ne $dstHash) {
    throw "Checksum mismatch after promote."
}

Write-Host "Promote completed."

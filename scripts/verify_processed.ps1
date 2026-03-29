param(
    [string]$ProjectRoot = (Get-Location).Path,
    [string]$DataDir = "processed",
    [int]$ExpectedCount = 2989,
    [long]$ExpectedBytes = 2860991576
)

$root = Resolve-Path -LiteralPath $ProjectRoot
$dataPath = Join-Path $root $DataDir

if (-not (Test-Path -LiteralPath $dataPath)) {
    throw "Data directory not found: $dataPath"
}

$measure = Get-ChildItem -LiteralPath $dataPath -File | Measure-Object -Property Length -Sum

Write-Host "Data directory: $dataPath"
Write-Host "File count: $($measure.Count)"
Write-Host "Total bytes: $($measure.Sum)"

if ($measure.Count -ne $ExpectedCount) {
    throw "File count mismatch. Expected $ExpectedCount, got $($measure.Count)."
}

if ($measure.Sum -ne $ExpectedBytes) {
    throw "Byte size mismatch. Expected $ExpectedBytes, got $($measure.Sum)."
}

Write-Host "Verification passed."

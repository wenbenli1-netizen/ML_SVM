param(
    [string]$ProjectRoot = (Get-Location).Path,
    [string]$DataDir = "processed",
    [string]$ZipName = "processed_dataset.zip"
)

$root = Resolve-Path -LiteralPath $ProjectRoot
$dataPath = Join-Path $root $DataDir
$zipPath = Join-Path $root $ZipName
$hashPath = "$zipPath.sha256.txt"
$manifestPath = Join-Path $root "processed_manifest.csv"

if (-not (Test-Path -LiteralPath $dataPath)) {
    throw "Data directory not found: $dataPath"
}

Write-Host "Packaging data from: $dataPath"

if (Test-Path -LiteralPath $zipPath) {
    Remove-Item -LiteralPath $zipPath -Force
}

Get-ChildItem -LiteralPath $dataPath -File |
    Select-Object Name, Length, LastWriteTime |
    Export-Csv -LiteralPath $manifestPath -NoTypeInformation -Encoding UTF8

tar.exe -a -cf $zipPath -C $root $DataDir

$hash = Get-FileHash -LiteralPath $zipPath -Algorithm SHA256
$hash.Hash | Set-Content -LiteralPath $hashPath -Encoding ascii

$measure = Get-ChildItem -LiteralPath $dataPath -File | Measure-Object -Property Length -Sum

Write-Host "Done."
Write-Host "Zip: $zipPath"
Write-Host "SHA256: $hashPath"
Write-Host "Manifest: $manifestPath"
Write-Host "Files: $($measure.Count)"
Write-Host "Bytes: $($measure.Sum)"

# Download a workspace-local language server; no system installation is needed.
# The older clangd bundled with VS 2022 cannot parse CUDA 13's runtime headers.
$ErrorActionPreference = 'Stop'
$clangdVersion = '22.1.6'
$clangdSha256 = 'ce54f16e0b4fd76d450eeda9664420b195360b73febcfe40e661108fa57f2ce1'
$projectRoot = Split-Path -Parent $PSScriptRoot
$toolsDirectory = Join-Path $projectRoot 'build/tools'
$clangdExecutable = Join-Path $toolsDirectory "clangd_$clangdVersion/bin/clangd.exe"

if (-not (Test-Path -LiteralPath $clangdExecutable)) {
    New-Item -ItemType Directory -Path $toolsDirectory -Force | Out-Null
    $archive = Join-Path $toolsDirectory "clangd-windows-$clangdVersion.zip"
    if (-not (Test-Path -LiteralPath $archive)) {
        $downloadUrl = "https://github.com/clangd/clangd/releases/download/$clangdVersion/clangd-windows-$clangdVersion.zip"
        Invoke-WebRequest -Uri $downloadUrl -OutFile $archive
    }
    if ((Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash -ne $clangdSha256) {
        throw "Unexpected clangd archive checksum: $archive"
    }
    Expand-Archive -LiteralPath $archive -DestinationPath $toolsDirectory -Force
}

& $clangdExecutable --version
if ($LASTEXITCODE -ne 0) { throw 'clangd did not start successfully.' }
Write-Host 'Install the recommended clangd extension, then run clangd: Restart language server in VS Code.'

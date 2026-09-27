<#
.SYNOPSIS
Configure, build, or test KiRaRay from ordinary Windows PowerShell.
.DESCRIPTION
Initializes the VS 2022 x64 build environment. Existing builds retain their
cached compiler; new builds use MSVC 14.44 unless Toolset is specified.
Build packages the Blender extension. Test builds all targets and runs CPU tests.
SDKs must already be installed; this script does not download dependencies.
.EXAMPLE
.\common\build\windows.ps1 -Preset blender -Action build
.EXAMPLE
.\common\build\windows.ps1 -Preset usd -Action configure -CMakeArgs '-DKRR_USD_ROOT=D:/SDK/OpenUSD'
#>
[CmdletBinding()]
param(
    [ValidateSet('blender', 'usd')]
    [string]$Preset = 'blender',
    [ValidateSet('configure', 'build', 'test')]
    [string]$Action = 'build',
    [ValidateRange(1, 256)]
    [int]$Jobs = 4,
    [string]$Toolset,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$CMakeArgs = @()
)

$ErrorActionPreference = 'Stop'
Write-Host "KiRaRay preset: $Preset; action: $Action"
$repository = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
$buildDirectory = if ($Preset -eq 'blender') { 'host' } else { 'standalone' }
$cache = Join-Path $repository "build/blender-interop/$buildDirectory/CMakeCache.txt"
$cachedCompiler = $null
$installation = $null

if (Test-Path -LiteralPath $cache) {
    $compilerEntry = Get-Content -LiteralPath $cache | Select-String '^CMAKE_CXX_COMPILER:(?:FILEPATH|STRING)=(.+)$' | Select-Object -First 1
    if ($compilerEntry) {
        $cachedCompiler = $compilerEntry.Matches[0].Groups[1].Value
        if ($cachedCompiler -notmatch '^(?<installation>.+)[\\/]VC[\\/]Tools[\\/]MSVC[\\/](?<version>[^\\/]+)[\\/]bin[\\/]Hostx64[\\/]x64[\\/]cl\.exe$') {
            throw "The preset build uses an unsupported compiler: $cachedCompiler. Use its original build environment or a separate build directory."
        }
        $installation = $Matches.installation
        $cachedToolset = $Matches.version
        if ($Toolset -and $cachedToolset -ne $Toolset -and -not $cachedToolset.StartsWith("$Toolset.")) {
            throw "The build cache uses MSVC $cachedToolset, but -Toolset selected $Toolset. Use the cached toolset or a separate build directory."
        }
        $Toolset = $cachedToolset
        if (-not (Test-Path -LiteralPath $cachedCompiler)) {
            throw "The cached compiler no longer exists: $cachedCompiler. Restore that toolset or configure a separate build directory."
        }
    }
}

if (-not $Toolset) { $Toolset = '14.44' }
if ($Toolset -notmatch '^14\.\d+(\.\d+)?$') {
    throw 'Toolset must be an MSVC version such as 14.44 or 14.44.35207.'
}
if (-not $installation) {
    $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio/Installer/vswhere.exe'
    if (-not (Test-Path -LiteralPath $vswhere)) {
        throw 'Install Visual Studio 2022 Build Tools with Desktop development with C++ (including MSVC 14.44 and the Windows SDK).'
    }
    $installation = & $vswhere -latest -products '*' -version '[17.0,18.0)' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    if ($LASTEXITCODE -ne 0 -or -not $installation) {
        throw 'No Visual Studio 2022 installation with the C++ build tools was found.'
    }
}

$module = Join-Path $installation 'Common7/Tools/Microsoft.VisualStudio.DevShell.dll'
if (-not (Test-Path -LiteralPath $module)) {
    throw "The Visual Studio developer shell is missing: $module"
}
Import-Module $module
Enter-VsDevShell -VsInstallPath $installation -SkipAutomaticLocation -DevCmdArguments "-arch=x64 -host_arch=x64 -vcvars_ver=$Toolset"
$compiler = Get-Command cl.exe -CommandType Application -ErrorAction Stop
if (-not $env:INCLUDE -or -not $env:LIB) {
    throw "Visual Studio did not initialize INCLUDE and LIB. Install MSVC $Toolset and the Windows SDK using Visual Studio Installer."
}
if ($cachedCompiler -and [IO.Path]::GetFullPath($compiler.Source) -ne [IO.Path]::GetFullPath($cachedCompiler)) {
    throw "The developer shell selected $($compiler.Source), but this build requires $cachedCompiler. Check the installed MSVC toolsets."
}
if ($compiler.Source -notmatch '[\\/]MSVC[\\/](?<version>[^\\/]+)[\\/]' -or
    ($Matches.version -ne $Toolset -and -not $Matches.version.StartsWith("$Toolset."))) {
    throw "The developer shell did not select MSVC $Toolset. Install that toolset using Visual Studio Installer."
}
$cmake = (Get-Command cmake.exe -CommandType Application -ErrorAction Stop).Source
$ctest = (Get-Command ctest.exe -CommandType Application -ErrorAction Stop).Source
$null = Get-Command ninja.exe -CommandType Application -ErrorAction Stop

function Invoke-BuildCommand([string]$Executable, [string[]]$Arguments) {
    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$([IO.Path]::GetFileName($Executable)) failed with exit code $LASTEXITCODE."
    }
}

Push-Location -LiteralPath $repository
try {
    Invoke-BuildCommand $cmake (@('--preset', $Preset) + $CMakeArgs)
    if ($Action -eq 'build') {
        $buildPreset = if ($Preset -eq 'blender') { 'blender-package' } else { 'usd' }
        Invoke-BuildCommand $cmake @('--build', '--preset', $buildPreset, '--parallel', "$Jobs")
    } elseif ($Action -eq 'test') {
        Invoke-BuildCommand $cmake @('--build', '--preset', $Preset, '--parallel', "$Jobs")
        if ($Preset -eq 'blender') {
            Invoke-BuildCommand $cmake @('--build', '--preset', 'blender-package', '--parallel', "$Jobs")
        }
        Invoke-BuildCommand $ctest @('--preset', "$Preset-cpu")
    }
} finally {
    Pop-Location
}

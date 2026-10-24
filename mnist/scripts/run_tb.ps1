# ================================================================
#
# Date  : Oct 6, 2026
# Author: Ishaan Parikh
#
# Runs an MNIST testbench with Icarus Verilog. Generates the test
# vectors if they are missing, compiles, then simulates.
#
# Usage (from any directory):
#   powershell -ExecutionPolicy Bypass -File mnist\scripts\run_tb.ps1
#   ... run_tb.ps1 -Tb winograd_3x3_conv -Dump
#   ... run_tb.ps1 -Regen    (rewrite the vector file first)
#
# ================================================================

param(
    [ValidateSet("convolve_3x3", "winograd_3x3_conv", "winograd_weight_transform")]
    [string]$Tb = "convolve_3x3",
    [switch]$Regen,
    [switch]$Dump
)

$ErrorActionPreference = "Stop"

$Iverilog = "C:\iverilog\bin\iverilog.exe"
$Vvp      = "C:\iverilog\bin\vvp.exe"

$Testbenches = @{
    "convolve_3x3"      = @{ Script  = "scripts\convolve_3x3_vectors.py"
                             Vectors = "tb\vectors\convolve_vectors.txt"
                             Rtl     = @("rtl\convolve_3x3.sv") }
    "winograd_3x3_conv" = @{ Script  = "scripts\winograd_3x3_vectors.py"
                             Vectors = "tb\vectors\winograd_vectors.txt"
                             Rtl     = @("rtl\winograd_3x3_conv.sv") }
    "winograd_weight_transform" = @{ Script  = "scripts\winograd_weight_transform_vectors.py"
                                     Vectors = "tb\vectors\weight_transform_vectors.txt"
                                     Rtl     = @("rtl\winograd_weight_transform.sv") }
}
$Config = $Testbenches[$Tb]

# The testbenches open tb\vectors\... relative to the working directory
Push-Location (Split-Path $PSScriptRoot -Parent)
try {
    if ($Regen -or -not (Test-Path $Config.Vectors)) {
        New-Item -ItemType Directory -Force "tb\vectors" | Out-Null
        Write-Host "Generating $($Config.Vectors)..."
        python $Config.Script
        if ($LASTEXITCODE -ne 0) { throw "Vector generation failed" }
    }

    $Output = "${Tb}_tb.vvp"
    & $Iverilog -g2012 -I . -o $Output @($Config.Rtl) "tb\${Tb}_tb.sv"
    if ($LASTEXITCODE -ne 0) { throw "Compile failed" }

    $SimArgs = @($Output)
    if ($Dump) { $SimArgs += "+dump" }
    & $Vvp @SimArgs
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}

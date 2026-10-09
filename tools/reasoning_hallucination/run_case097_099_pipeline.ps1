param(
    [Parameter(Mandatory = $true)][string]$DataDir,
    [Parameter(Mandatory = $true)][string]$CasesDir,
    [string]$OutputDir = "results/reasoning_quality"
)

python reasoning_quality_main.py --data-dir $DataDir --cases-dir $CasesDir --output-dir $OutputDir
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

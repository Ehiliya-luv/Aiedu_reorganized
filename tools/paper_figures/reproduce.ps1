param(
    [string]$Only = "",
    [string]$OutputDir = "results/paper_figures"
)

$arguments = @("paper_reproduction_main.py", "--output-dir", $OutputDir)
if ($Only) { $arguments += @("--only", $Only) }
python @arguments
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

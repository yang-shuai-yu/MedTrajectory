param(
    [switch]$Strict
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
Push-Location $projectRoot
try {
    python scripts/build_result_registry.py
    if ($Strict) {
        python scripts/audit_result_registry.py --strict
    } else {
        python scripts/audit_result_registry.py
    }
} finally {
    Pop-Location
}

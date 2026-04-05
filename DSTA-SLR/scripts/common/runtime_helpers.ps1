function Get-DstaSlrRoot {
    param(
        [string]$StartPath = $PSScriptRoot
    )

    $current = (Resolve-Path $StartPath).Path
    while ($true) {
        if ((Test-Path (Join-Path $current "main.py")) -and (Test-Path (Join-Path $current "scripts"))) {
            return $current
        }

        $parent = Split-Path -Parent $current
        if ([string]::Equals($parent, $current, [System.StringComparison]::OrdinalIgnoreCase)) {
            throw "Could not locate the DSTA-SLR repository root from $StartPath"
        }
        $current = $parent
    }
}

function Resolve-CondaExecutable {
    param(
        [switch]$AllowMissing
    )

    $candidates = @(
        $env:DSTA_SLR_CONDA_EXE,
        $env:CONDA_EXE
    )

    foreach ($commandName in @("conda.exe", "conda", "conda.bat")) {
        $command = Get-Command $commandName -ErrorAction SilentlyContinue
        if ($null -ne $command -and $command.Path) {
            $candidates += $command.Path
        }
    }

    if ($env:USERPROFILE) {
        $candidates += @(
            (Join-Path $env:USERPROFILE "miniconda3\Scripts\conda.exe"),
            (Join-Path $env:USERPROFILE "anaconda3\Scripts\conda.exe")
        )
    }

    foreach ($candidate in $candidates) {
        if ($candidate -and (Test-Path $candidate)) {
            return (Resolve-Path $candidate).Path
        }
    }

    if ($AllowMissing) {
        return $null
    }

    throw "Could not locate conda.exe. Set DSTA_SLR_CONDA_EXE or add Conda to PATH."
}

function Get-CondaBase {
    param(
        [string]$CondaExe = $(Resolve-CondaExecutable)
    )

    $condaBase = (& $CondaExe info --base | Select-Object -Last 1).Trim()
    if (-not $condaBase) {
        throw "Could not determine the Conda base directory from $CondaExe"
    }
    return $condaBase
}

function Resolve-DstaSlrPython {
    param(
        [string]$Root
    )

    if ($env:DSTA_SLR_PYTHON -and (Test-Path $env:DSTA_SLR_PYTHON)) {
        return (Resolve-Path $env:DSTA_SLR_PYTHON).Path
    }

    $rootPath = (Resolve-Path $Root).Path
    $condaExe = Resolve-CondaExecutable -AllowMissing
    if ($condaExe) {
        $condaBase = Get-CondaBase -CondaExe $condaExe
        $condaEnvPython = Join-Path $condaBase "envs\dsta-slr\python.exe"
        if (Test-Path $condaEnvPython) {
            return (Resolve-Path $condaEnvPython).Path
        }
    }

    $venvPython = Join-Path $rootPath ".venv\Scripts\python.exe"
    if (Test-Path $venvPython) {
        return (Resolve-Path $venvPython).Path
    }

    if ($env:CONDA_PREFIX) {
        $condaPrefixPython = Join-Path $env:CONDA_PREFIX "python.exe"
        if (Test-Path $condaPrefixPython) {
            return (Resolve-Path $condaPrefixPython).Path
        }
    }

    foreach ($commandName in @("python.exe", "python")) {
        $command = Get-Command $commandName -ErrorAction SilentlyContinue
        if ($null -ne $command -and $command.Path) {
            return (Resolve-Path $command.Path).Path
        }
    }

    throw "Could not locate a usable Python executable. Set DSTA_SLR_PYTHON or activate an environment first."
}

function Initialize-CondaShell {
    param(
        [string]$CondaExe = $(Resolve-CondaExecutable)
    )

    $hook = & $CondaExe shell.powershell hook | Out-String
    if (-not $hook.Trim()) {
        throw "Failed to initialize the Conda PowerShell hook from $CondaExe"
    }
    Invoke-Expression $hook
}

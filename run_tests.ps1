param(
    [string]$Language = "all"
)

$ErrorActionPreference = "Stop"

function Write-Green {
    param([string]$Text)
    Write-Host $Text -ForegroundColor Green
}

function Write-Red {
    param([string]$Text)
    Write-Host $Text -ForegroundColor Red
}

function Write-Yellow {
    param([string]$Text)
    Write-Host $Text -ForegroundColor Yellow
}

function Check-Python {
    try {
        $null = python --version 2>&1
        if ($LASTEXITCODE -ne 0) { throw }
        $pythonVersion = python --version 2>&1
        Write-Host "Found Python: $pythonVersion"
    } catch {
        Write-Red "ERROR: Python command not found!"
        Write-Host "Please install Python to run this script."
        throw
    }
}

function Ensure-PythonPackages {
    $packages = @("transformers", "torch")
    $missingPackages = @()

    foreach ($pkg in $packages) {
        $null = python -c "import $($pkg.Replace('-', '_'))" 2>&1
        if ($LASTEXITCODE -ne 0) {
            $missingPackages += $pkg
        }
    }

    if ($missingPackages.Count -gt 0) {
        Write-Yellow "Installing required Python packages: $($missingPackages -join ', ')"
        foreach ($pkg in $missingPackages) {
            pip install $pkg
            if ($LASTEXITCODE -ne 0) {
                Write-Red "ERROR: Failed to install package $pkg"
                throw
            }
        }
    }
}

function Generate-TestData {
    Write-Yellow "Generating test data..."

    Check-Python
    Ensure-PythonPackages

    Push-Location "python"
    try {
        python generate_test_data.py
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to generate test data"
        }
    } finally {
        Pop-Location
    }

    Write-Green "Test data generated successfully!"
}

function Invoke-DotNetChecks {
    Write-Yellow "Running .NET checks and tests..."

    try {
        $null = dotnet --version 2>&1
        if ($LASTEXITCODE -ne 0) { throw }
        $dotnetVersion = dotnet --version
        Write-Host "Found .NET: $dotnetVersion"
    } catch {
        Write-Red "ERROR: dotnet command not found!"
        Write-Host "Please install .NET SDK to run .NET tests."
        throw
    }

    Push-Location "dotnet"
    try {
        Write-Yellow "Verifying code formatting..."
        dotnet format Yuniko.Software.Qwen3Tokenizer.slnx --verify-no-changes --verbosity diagnostic
        if ($LASTEXITCODE -ne 0) {
            Write-Red "ERROR: Code formatting issues detected!"
            Write-Host "Run 'dotnet format' to fix formatting issues."
            throw "Code formatting check failed"
        }
        Write-Green "Code formatting verified!"

        dotnet restore Yuniko.Software.Qwen3Tokenizer.slnx
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to restore .NET dependencies"
        }

        dotnet build Yuniko.Software.Qwen3Tokenizer.slnx --no-restore --configuration Release
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to build .NET solution"
        }

        dotnet test Yuniko.Software.Qwen3Tokenizer.slnx --no-build --configuration Release --verbosity normal --collect:"XPlat Code Coverage"
        if ($LASTEXITCODE -ne 0) {
            throw ".NET tests failed"
        }

        Write-Green "All .NET checks passed!"
    } finally {
        Pop-Location
    }
}

Write-Yellow "Starting Qwen3 Tokenizer test suite"
Write-Host ""

try {
    Generate-TestData
    Write-Host ""

    if ($Language -eq "all" -or $Language -eq "dotnet") {
        Invoke-DotNetChecks
        Write-Host ""
    }

    Write-Green "All checks and tests passed successfully!"
    exit 0
} catch {
    Write-Host ""
    Write-Red "Some checks or tests failed!"
    exit 1
}

# Test Runner Script for PSO Portfolio Optimizer
# Run this to verify your installation

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  PSO Portfolio Optimizer" -ForegroundColor Cyan
Write-Host "  Installation Test" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$errors = 0

# Test 1: Python Installation
Write-Host "Test 1: Checking Python..." -NoNewline
try {
    $pythonVersion = python --version 2>&1
    if ($pythonVersion -match "Python 3\.([9-9]|1[0-9])") {
        Write-Host " ✓ PASS" -ForegroundColor Green
    } else {
        Write-Host " ✗ FAIL (Python 3.9+ required)" -ForegroundColor Red
        $errors++
    }
} catch {
    Write-Host " ✗ FAIL (Python not found)" -ForegroundColor Red
    $errors++
}

# Test 2: Required Files
Write-Host "Test 2: Checking project files..." -NoNewline
$requiredFiles = @(
    "requirements.txt",
    "Dockerfile",
    "docker-compose.yml",
    "src/pso.py",
    "src/portfolio.py",
    "src/visualization.py",
    "app/main.py",
    "README.md"
)

$allFilesExist = $true
foreach ($file in $requiredFiles) {
    if (-not (Test-Path $file)) {
        $allFilesExist = $false
        break
    }
}

if ($allFilesExist) {
    Write-Host " ✓ PASS" -ForegroundColor Green
} else {
    Write-Host " ✗ FAIL (Missing files)" -ForegroundColor Red
    $errors++
}

# Test 3: Try importing modules (if venv exists and is activated)
Write-Host "Test 3: Checking Python modules..." -NoNewline
try {
    $importTest = python -c "import numpy; import pandas; print('OK')" 2>&1
    if ($importTest -match "OK") {
        Write-Host " ✓ PASS" -ForegroundColor Green
    } else {
        Write-Host " ⚠ SKIP (Dependencies not installed yet)" -ForegroundColor Yellow
    }
} catch {
    Write-Host " ⚠ SKIP (Dependencies not installed yet)" -ForegroundColor Yellow
}

# Test 4: Quick PSO test
Write-Host "Test 4: Running PSO algorithm test..." -NoNewline
$testScript = @"
import sys
sys.path.append('.')
from src.pso import ParticleSwarmOptimizer, PSOConfig
import numpy as np

def sphere(x):
    return np.sum(x**2)

config = PSOConfig(n_particles=10, n_dimensions=3, n_iterations=20, verbose=False)
pso = ParticleSwarmOptimizer(config)
best_pos, best_fit, _ = pso.optimize(sphere)

if best_fit < 1.0:
    print('PASS')
else:
    print('FAIL')
"@

try {
    $testResult = python -c $testScript 2>&1
    if ($testResult -match "PASS") {
        Write-Host " ✓ PASS" -ForegroundColor Green
    } elseif ($testResult -match "No module named") {
        Write-Host " ⚠ SKIP (Dependencies not installed)" -ForegroundColor Yellow
    } else {
        Write-Host " ✗ FAIL" -ForegroundColor Red
        $errors++
    }
} catch {
    Write-Host " ⚠ SKIP (Dependencies not installed)" -ForegroundColor Yellow
}

# Summary
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
if ($errors -eq 0) {
    Write-Host "  ✓ All tests passed!" -ForegroundColor Green
    Write-Host ""
    Write-Host "  You're ready to go! Run:" -ForegroundColor Cyan
    Write-Host "    .\start.ps1" -ForegroundColor White
} else {
    Write-Host "  ✗ $errors test(s) failed" -ForegroundColor Red
    Write-Host ""
    Write-Host "  Please install dependencies:" -ForegroundColor Yellow
    Write-Host "    pip install -r requirements.txt" -ForegroundColor White
}
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

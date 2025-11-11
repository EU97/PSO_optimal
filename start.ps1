# PSO Portfolio Optimizer - Quick Start Script
# This script will help you get started with the PSO Portfolio Optimizer

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  PSO Portfolio Optimizer" -ForegroundColor Cyan
Write-Host "  Quick Start Script" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check Python installation
Write-Host "Checking Python installation..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version 2>&1
    Write-Host "✓ Python found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "✗ Python not found. Please install Python 3.9 or higher." -ForegroundColor Red
    Write-Host "  Download from: https://www.python.org/downloads/" -ForegroundColor Yellow
    exit 1
}

Write-Host ""
Write-Host "Choose your preferred setup method:" -ForegroundColor Yellow
Write-Host "1. Docker (Recommended - Isolated environment)" -ForegroundColor White
Write-Host "2. Local Installation (Direct Python environment)" -ForegroundColor White
Write-Host ""

$choice = Read-Host "Enter your choice (1 or 2)"

if ($choice -eq "1") {
    Write-Host ""
    Write-Host "Setting up with Docker..." -ForegroundColor Cyan
    
    # Check Docker installation
    Write-Host "Checking Docker installation..." -ForegroundColor Yellow
    try {
        $dockerVersion = docker --version 2>&1
        Write-Host "✓ Docker found: $dockerVersion" -ForegroundColor Green
    } catch {
        Write-Host "✗ Docker not found. Please install Docker Desktop." -ForegroundColor Red
        Write-Host "  Download from: https://www.docker.com/products/docker-desktop" -ForegroundColor Yellow
        exit 1
    }
    
    Write-Host ""
    Write-Host "Building and starting Docker containers..." -ForegroundColor Yellow
    docker-compose up --build -d
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host ""
        Write-Host "✓ Application started successfully!" -ForegroundColor Green
        Write-Host ""
        Write-Host "Access the application at: http://localhost:8501" -ForegroundColor Cyan
        Write-Host ""
        Write-Host "Useful commands:" -ForegroundColor Yellow
        Write-Host "  Stop:     docker-compose stop" -ForegroundColor White
        Write-Host "  Restart:  docker-compose restart" -ForegroundColor White
        Write-Host "  Logs:     docker-compose logs -f" -ForegroundColor White
        Write-Host "  Remove:   docker-compose down" -ForegroundColor White
        Write-Host ""
        
        # Open browser
        Start-Process "http://localhost:8501"
    } else {
        Write-Host "✗ Docker setup failed. Check the error messages above." -ForegroundColor Red
    }
    
} elseif ($choice -eq "2") {
    Write-Host ""
    Write-Host "Setting up local installation..." -ForegroundColor Cyan
    
    # Check if venv exists
    if (Test-Path "venv") {
        Write-Host "Virtual environment already exists." -ForegroundColor Yellow
        $recreate = Read-Host "Recreate virtual environment? (y/n)"
        if ($recreate -eq "y") {
            Write-Host "Removing existing virtual environment..." -ForegroundColor Yellow
            Remove-Item -Recurse -Force venv
        }
    }
    
    # Create virtual environment
    if (-not (Test-Path "venv")) {
        Write-Host "Creating virtual environment..." -ForegroundColor Yellow
        python -m venv venv
        Write-Host "✓ Virtual environment created" -ForegroundColor Green
    }
    
    # Activate virtual environment
    Write-Host "Activating virtual environment..." -ForegroundColor Yellow
    & .\venv\Scripts\Activate.ps1
    
    # Upgrade pip
    Write-Host "Upgrading pip..." -ForegroundColor Yellow
    python -m pip install --upgrade pip --quiet
    
    # Install requirements
    Write-Host "Installing dependencies..." -ForegroundColor Yellow
    pip install -r requirements.txt --quiet
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ Dependencies installed successfully!" -ForegroundColor Green
        Write-Host ""
        Write-Host "Starting application..." -ForegroundColor Yellow
        Write-Host ""
        
        # Start Streamlit
        streamlit run app/main.py
    } else {
        Write-Host "✗ Installation failed. Check the error messages above." -ForegroundColor Red
    }
    
} else {
    Write-Host "Invalid choice. Please run the script again and choose 1 or 2." -ForegroundColor Red
    exit 1
}

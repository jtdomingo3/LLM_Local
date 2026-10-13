<#
.SYNOPSIS
    One-click Ollama setup script for LLM Local Microservice.
    Sets all optimizations from our research and pulls required models.

.DESCRIPTION
    This script:
    1. Sets Windows environment variables for Flash Attention & KV Cache Quantization
    2. Checks if Ollama is installed and running
    3. Pulls the three optimized models (phi4-mini, qwen2.5-coder:3b, qwen2.5-coder:7b)
    4. Verifies GPU detection and optimization flags
#>

Write-Host ""
Write-Host "=============================================================" -ForegroundColor Cyan
Write-Host "  LLM Local Microservice - Ollama Setup" -ForegroundColor Cyan
Write-Host "  Applying ALL optimizations from research" -ForegroundColor Cyan
Write-Host "=============================================================" -ForegroundColor Cyan
Write-Host ""

# ---------------------------------------------------------------
# Step 1: Set Ollama optimization environment variables (User level)
# ---------------------------------------------------------------
Write-Host "[1/5] Setting optimization environment variables..." -ForegroundColor Yellow

# Flash Attention - free speed boost, zero quality loss
[System.Environment]::SetEnvironmentVariable("OLLAMA_FLASH_ATTENTION", "1", "User")
Write-Host "  + OLLAMA_FLASH_ATTENTION = 1 (10-15% speed boost)" -ForegroundColor Green

# KV Cache Quantization - 50% less cache memory, negligible quality loss
[System.Environment]::SetEnvironmentVariable("OLLAMA_KV_CACHE_TYPE", "q8_0", "User")
Write-Host "  + OLLAMA_KV_CACHE_TYPE = q8_0 (50% KV cache reduction)" -ForegroundColor Green

# Aggressive VRAM cleanup & single-model enforcement (essential for 4GB VRAM)
[System.Environment]::SetEnvironmentVariable("OLLAMA_KEEP_ALIVE", "90s", "User")
Write-Host "  + OLLAMA_KEEP_ALIVE = 90s (releases VRAM after 90s idle)" -ForegroundColor Green

[System.Environment]::SetEnvironmentVariable("OLLAMA_MAX_LOADED_MODELS", "1", "User")
Write-Host "  + OLLAMA_MAX_LOADED_MODELS = 1 (prevents VRAM spillover)" -ForegroundColor Green

[System.Environment]::SetEnvironmentVariable("OLLAMA_NUM_PARALLEL", "1", "User")
Write-Host "  + OLLAMA_NUM_PARALLEL = 1 (prevents multi-context VRAM surge)" -ForegroundColor Green

# Also set for current session
$env:OLLAMA_FLASH_ATTENTION = "1"
$env:OLLAMA_KV_CACHE_TYPE = "q8_0"
$env:OLLAMA_KEEP_ALIVE = "90s"
$env:OLLAMA_MAX_LOADED_MODELS = "1"
$env:OLLAMA_NUM_PARALLEL = "1"

Write-Host "  Environment variables set (User level + current session)" -ForegroundColor Green
Write-Host ""

# ---------------------------------------------------------------
# Step 2: Check if Ollama is installed
# ---------------------------------------------------------------
Write-Host "[2/5] Checking Ollama installation..." -ForegroundColor Yellow

$ollamaPath = Get-Command ollama -ErrorAction SilentlyContinue
if (-not $ollamaPath) {
    Write-Host "  ! Ollama is NOT installed." -ForegroundColor Red
    Write-Host "  Please download and install from: https://ollama.com/download" -ForegroundColor Red
    Write-Host "  After installing, re-run this script." -ForegroundColor Red
    Write-Host ""
    Read-Host "Press Enter to exit"
    exit 1
}
Write-Host "  + Ollama found at: $($ollamaPath.Source)" -ForegroundColor Green
Write-Host ""

# ---------------------------------------------------------------
# Step 3: Check if Ollama server is running
# ---------------------------------------------------------------
Write-Host "[3/5] Checking Ollama server status..." -ForegroundColor Yellow

try {
    $response = Invoke-RestMethod -Uri "http://localhost:11434/api/tags" -Method Get -TimeoutSec 5 -ErrorAction Stop
    Write-Host "  + Ollama server is running" -ForegroundColor Green
    
    # List existing models
    if ($response.models -and $response.models.Count -gt 0) {
        Write-Host "  + Currently installed models:" -ForegroundColor Green
        foreach ($model in $response.models) {
            $sizeGB = [math]::Round($model.size / 1GB, 2)
            Write-Host "    - $($model.name) ($sizeGB GB)" -ForegroundColor Gray
        }
    }
} catch {
    Write-Host "  ! Ollama server is not running. Starting it..." -ForegroundColor Yellow
    Start-Process -FilePath "ollama" -ArgumentList "serve" -WindowStyle Hidden
    Start-Sleep -Seconds 3
    
    try {
        Invoke-RestMethod -Uri "http://localhost:11434/api/tags" -Method Get -TimeoutSec 5 -ErrorAction Stop | Out-Null
        Write-Host "  + Ollama server started successfully" -ForegroundColor Green
    } catch {
        Write-Host "  ! Failed to start Ollama server. Please start it manually." -ForegroundColor Red
        Write-Host "    Run: ollama serve" -ForegroundColor Red
        Read-Host "Press Enter to exit"
        exit 1
    }
}
Write-Host ""

# ---------------------------------------------------------------
# Step 4: Pull required models
# ---------------------------------------------------------------
Write-Host "[4/5] Pulling optimized models..." -ForegroundColor Yellow
Write-Host "  This may take several minutes depending on your internet speed." -ForegroundColor Gray
Write-Host ""

$models = @(
    @{ Name = "phi4-mini";           Desc = "Daily driver (3.8B, ~2.5 GB) - reasoning, chat, summarization" },
    @{ Name = "qwen2.5-coder:3b";   Desc = "Coding assistant (3B, ~2.0 GB) - fast code generation" },
    @{ Name = "qwen2.5-coder:7b";   Desc = "Power coding (7B, ~4.5 GB) - complex code, hybrid GPU/CPU" }
)

foreach ($model in $models) {
    Write-Host "  Pulling $($model.Name)..." -ForegroundColor Cyan
    Write-Host "    $($model.Desc)" -ForegroundColor Gray
    
    & ollama pull $model.Name
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "    + Done!" -ForegroundColor Green
    } else {
        Write-Host "    ! Failed to pull $($model.Name). You can try manually: ollama pull $($model.Name)" -ForegroundColor Red
    }
    Write-Host ""
}

# ---------------------------------------------------------------
# Step 5: Verify setup
# ---------------------------------------------------------------
Write-Host "[5/5] Verifying setup..." -ForegroundColor Yellow

# Check GPU
Write-Host "  Checking GPU..." -ForegroundColor Gray
try {
    $nvidiaSmi = & nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>$null
    if ($nvidiaSmi) {
        Write-Host "  + GPU detected: $nvidiaSmi" -ForegroundColor Green
    }
} catch {
    Write-Host "  ! nvidia-smi not found. GPU acceleration may not work." -ForegroundColor Yellow
}

# Verify environment variables
Write-Host "  Checking optimization flags..." -ForegroundColor Gray
$flashAttn = [System.Environment]::GetEnvironmentVariable("OLLAMA_FLASH_ATTENTION", "User")
$kvCache = [System.Environment]::GetEnvironmentVariable("OLLAMA_KV_CACHE_TYPE", "User")

if ($flashAttn -eq "1") {
    Write-Host "  + Flash Attention: ENABLED" -ForegroundColor Green
} else {
    Write-Host "  ! Flash Attention: NOT SET" -ForegroundColor Red
}

if ($kvCache -eq "q8_0") {
    Write-Host "  + KV Cache Quantization: q8_0 (50% savings)" -ForegroundColor Green
} else {
    Write-Host "  ! KV Cache Quantization: NOT SET" -ForegroundColor Red
}

$keepAlive = [System.Environment]::GetEnvironmentVariable("OLLAMA_KEEP_ALIVE", "User")
if ($keepAlive) {
    Write-Host "  + Keep Alive: $keepAlive" -ForegroundColor Green
} else {
    Write-Host "  ! Keep Alive: NOT SET" -ForegroundColor Yellow
}

$maxLoaded = [System.Environment]::GetEnvironmentVariable("OLLAMA_MAX_LOADED_MODELS", "User")
if ($maxLoaded -eq "1") {
    Write-Host "  + Max Loaded Models: 1 (safe for 4GB VRAM)" -ForegroundColor Green
} else {
    Write-Host "  ! Max Loaded Models: NOT SET" -ForegroundColor Yellow
}

# Final summary
Write-Host ""
Write-Host "=============================================================" -ForegroundColor Cyan
Write-Host "  Setup Complete!" -ForegroundColor Green
Write-Host "=============================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "  IMPORTANT: You must RESTART Ollama for env vars to take effect." -ForegroundColor Yellow
Write-Host "  Right-click the Ollama tray icon -> Quit, then relaunch it." -ForegroundColor Yellow
Write-Host ""
Write-Host "  To start the microservice:" -ForegroundColor Gray
Write-Host "    cd $(Get-Location)" -ForegroundColor White
Write-Host "    pip install -r requirements.txt" -ForegroundColor White
Write-Host "    python run.py" -ForegroundColor White
Write-Host ""
Write-Host "  API will be available at: http://localhost:8100" -ForegroundColor Green
Write-Host "  Swagger docs at:          http://localhost:8100/docs" -ForegroundColor Green
Write-Host ""

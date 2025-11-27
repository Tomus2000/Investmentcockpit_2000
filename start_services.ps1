# Investment Cockpit - Start All Services
# This script starts both the Streamlit app and Telegram bot simultaneously

Write-Host "🚀 Starting Investment Cockpit Services..." -ForegroundColor Green

# Start Streamlit app in background
Write-Host "📊 Starting Streamlit Dashboard..." -ForegroundColor Yellow
Start-Process -NoNewWindow -FilePath "py" -ArgumentList "-m", "streamlit", "run", "app.py", "--server.headless=true", "--browser.gatherUsageStats=false"

# Wait a moment for Streamlit to start
Start-Sleep -Seconds 3

# Start Telegram bot in background
Write-Host "🤖 Starting Telegram Bot..." -ForegroundColor Yellow
Start-Process -NoNewWindow -FilePath "py" -ArgumentList "bot.py"

Write-Host "✅ Both services are now running!" -ForegroundColor Green
Write-Host "📊 Streamlit Dashboard: http://localhost:8501" -ForegroundColor Cyan
Write-Host "🤖 Telegram Bot: Active and monitoring your portfolio" -ForegroundColor Cyan
Write-Host "" -ForegroundColor White
Write-Host "Press Ctrl+C to stop all services" -ForegroundColor Red

# Keep script running to maintain services
try {
    while ($true) {
        Start-Sleep -Seconds 10
        # Check if processes are still running
        $streamlit = Get-Process -Name "python" -ErrorAction SilentlyContinue | Where-Object {$_.CommandLine -like "*streamlit*"}
        $bot = Get-Process -Name "python" -ErrorAction SilentlyContinue | Where-Object {$_.CommandLine -like "*bot.py*"}
        
        if (-not $streamlit) {
            Write-Host "⚠️ Streamlit stopped, restarting..." -ForegroundColor Yellow
            Start-Process -NoNewWindow -FilePath "py" -ArgumentList "-m", "streamlit", "run", "app.py", "--server.headless=true", "--browser.gatherUsageStats=false"
        }
        
        if (-not $bot) {
            Write-Host "⚠️ Telegram bot stopped, restarting..." -ForegroundColor Yellow
            Start-Process -NoNewWindow -FilePath "py" -ArgumentList "bot.py"
        }
    }
} catch {
    Write-Host "🛑 Stopping all services..." -ForegroundColor Red
    # Kill all Python processes related to our services
    Get-Process -Name "python" -ErrorAction SilentlyContinue | Where-Object {$_.CommandLine -like "*streamlit*" -or $_.CommandLine -like "*bot.py*"} | Stop-Process -Force
}


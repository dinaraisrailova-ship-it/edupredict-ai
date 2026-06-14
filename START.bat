@echo off
title EduPredict AI - 24/7 Tunnel
color 0B
cd /d "%~dp0"

echo.
echo  ================================================
echo   EDUPREDICT AI - ISHGA TUSHIRILMOQDA
echo  ================================================
echo.

:: Eski jarayonlarni to'xtatish
taskkill /F /IM cloudflared.exe >nul 2>&1

:: Streamlit ishga tushirish
echo  [1/3] Ilova ishga tushirilmoqda...
start "Streamlit" /MIN cmd /c "python -m streamlit run app.py"
timeout /t 15 /nobreak >nul

:: Tunnel ishga tushirish va URL olish
echo  [2/3] Internet tunnel ochilmoqda...
start "Tunnel" /MIN cmd /c ""%~dp0..\cloudflared.exe" tunnel --protocol http2 --url http://localhost:8502 > "%~dp0tunnel_log.txt" 2>&1"
timeout /t 18 /nobreak >nul

:: URL ni olish
echo  [3/3] Link tayyorlanmoqda...
python "%~dp0update_link.py"

echo.
echo  ================================================
echo   TAYYOR! Link yuqorida ko'rsatildi
echo  ================================================
echo.
pause

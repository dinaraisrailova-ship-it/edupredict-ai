@echo off
title EduPredict AI - Internet Tunnel
color 0B

echo.
echo  ================================================
echo   EduPredict AI -- Internet orqali ulash
echo   Telefon + Noutbuk + Har qanday Wi-Fi
echo  ================================================
echo.

:: Eski jarayonlarni to'xtatish
taskkill /F /IM cloudflared.exe >nul 2>&1

:: Streamlit ishga tushirish
echo  [1/2] Streamlit ishga tushirilmoqda (port 8502)...
start "EduPredict Streamlit" /MIN cmd /c "cd /d %~dp0 && python -m streamlit run app.py"
echo  Yuklanmoqda...
timeout /t 12 /nobreak >nul

:: Cloudflare Tunnel
echo  [2/2] Internet tunnel ochilmoqda...
echo.
echo  Quyida ko'rinadigan https:// manzilni
echo  telefon yoki boshqa noutbukda brauzerga kiriting:
echo.
echo  ================================================
echo.

"%~dp0..\cloudflared.exe" tunnel --url http://localhost:8502

echo.
echo  Tunnel yopildi. Oyna yoping.
pause

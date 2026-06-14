@echo off
title EduPredict AI - Internet Tunnel
color 0B

echo.
echo  ================================================
echo   EduPredict AI -- Internet orqali ulash
echo  ================================================
echo.

:: Streamlit ishga tushirish (agar ishlamayotgan bo'lsa)
echo  [1/2] Streamlit ishga tushirilmoqda...
start "EduPredict Streamlit" /MIN cmd /c "cd /d %~dp0 && python -m streamlit run app.py --server.address 0.0.0.0 --server.port 8501 --server.enableCORS false --server.enableXsrfProtection false --server.headless true"

:: 8 soniya kutish
echo  Yuklanmoqda, kuting...
timeout /t 8 /nobreak >nul

:: Cloudflare Tunnel ochish
echo  [2/2] Internet tunnel ochilmoqda...
echo.
echo  Quyidagi URLni boshqa noutbukda brauzerga kiriting:
echo  (URL bir necha soniyadan so'ng chiqadi)
echo.
echo  ================================================

"%~dp0..\cloudflared.exe" tunnel --url http://localhost:8501

echo.
echo  Tunnel yopildi.
pause

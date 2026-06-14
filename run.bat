@echo off
title EduPredict AI - Student Dropout Prediction
color 0B

echo.
echo  ================================================
echo   EduPredict AI -- Talabalar Dropout Bashorati
echo  ================================================
echo.

:: Firewall portini ochish (administrator kerak emas)
netsh advfirewall firewall add rule name="Streamlit EduPredict 8501" dir=in action=allow protocol=TCP localport=8501 >nul 2>&1

:: IP manzilni aniqlash
for /f "tokens=2 delims=:" %%a in ('ipconfig ^| findstr /i "IPv4" ^| findstr /v "169.254"') do (
    set IP=%%a
    goto :found
)
:found
set IP=%IP: =%

echo  Mahalliy URL  : http://localhost:8501
echo  Tarmoq URL    : http://%IP%:8501
echo.
echo  Boshqa kompyuterdan ochish uchun:
echo  http://%IP%:8501
echo.
echo  (Bir xil Wi-Fi tarmog'ida bo'lishi kerak)
echo  ================================================
echo.

cd /d "%~dp0"
python -m streamlit run app.py --server.address 0.0.0.0 --server.port 8501 --browser.gatherUsageStats false

pause

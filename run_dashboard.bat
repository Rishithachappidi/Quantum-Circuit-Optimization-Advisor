@echo off
echo ============================================
echo Quantum Circuit Optimization Advisor
echo ============================================
echo.

call conda activate myQisk

if errorlevel 1 (
    echo Could not activate myQisk.
    echo Open Anaconda Prompt and run:
    echo conda activate myQisk
    echo streamlit run app.py
    pause
    exit /b 1
)

streamlit run app.py
pause

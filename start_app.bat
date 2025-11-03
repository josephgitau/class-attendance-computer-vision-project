@echo off
echo Starting Student Attendance System...
echo.

:: Check if streamlit is installed
python -c "import streamlit" 2>nul
if errorlevel 1 (
    echo ERROR: Streamlit is not installed.
    echo Please run: pip install streamlit
    pause
    exit /b 1
)

:: Check if ultralytics is installed
python -c "import ultralytics" 2>nul
if errorlevel 1 (
    echo ERROR: Ultralytics is not installed.
    echo Please run: pip install ultralytics
    pause
    exit /b 1
)

:: Set environment variables to suppress warnings
set PYTHONWARNINGS=ignore
set TORCH_LOGS=

echo All dependencies are installed.
echo Starting the web application...
echo.
echo The application will open in your browser at: http://localhost:8501
echo Press Ctrl+C to stop the application
echo.

:: Start streamlit
streamlit run app.py --server.port 8501 --server.address localhost

pause
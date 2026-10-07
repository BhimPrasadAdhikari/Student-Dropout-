@echo off
echo =====================================================================
echo Launching Student Dropout Early-Warning & Prediction System (Streamlit)
echo CS378 - Data Mining & Data Warehousing Lab
echo C.V. Raman Global University, Bhubaneswar
echo =====================================================================
call .\.venv\Scripts\activate.bat
streamlit run app.py
pause

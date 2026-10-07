# Student-Dropout-

An intelligent early-warning decision-support system that predicts student dropout risks from lecture attendance patterns and multi-semester academic progression records. The system utilizes machine learning classifiers (trained on 4,424 student records) and a dimensional data warehouse architecture to identify at-risk students in real-time, providing actionable institutional intervention protocols.

---

## Setup and Execution

### 1. Clone the Repository
```bash
git clone https://github.com/BhimPrasadAdhikari/Student-Dropout-.git
cd Student-Dropout-
```

### 2. Create and Activate Virtual Environment
```bash
# Windows
python -m venv .venv
.\.venv\Scripts\activate

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Prediction Console
```bash
streamlit run app.py
```
*(On Windows, you can also double-click `run_app.bat`)*

Open your browser at `http://localhost:8501`.

---

## Project Structure

```
Student-Dropout-/
│
├── .streamlit/
│   └── config.toml          # Streamlit theme & UI configuration
│
├── app.py                   # Streamlit Early-Warning & Risk Prediction Console
├── train_models.py          # ML training, cross-validation & evaluation pipeline
├── run_app.bat              # One-click Windows application launcher
├── requirements.txt         # Python package dependencies
├── README.md                # Project documentation
│
├── data/
│   ├── data.csv             # Raw student dataset (4,424 records)
│   └── enriched_student_data.csv
│
├── models/
│   ├── best_model_bundle.joblib
│   ├── scaler.joblib
│   ├── feature_names.joblib
│   └── model_comparison.csv
│
└── figures/
    ├── star_schema_dwh.png
    ├── eda_target_distribution.png
    ├── attendance_vs_dropout_kde.png
    ├── gpa_vs_attendance_scatter.png
    ├── correlation_heatmap.png
    ├── model_accuracy_comparison.png
    ├── roc_curves_all_models.png
    ├── confusion_matrix_best.png
    └── feature_importance.png
```

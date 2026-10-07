# Student-Dropout-

## Predicting Student Dropout from Attendance and Academic Records
**Course:** Data Mining and Data Warehousing Lab (`CS378`) • Credits: `(0-3-0-0)2`  
**Institution:** C.V. Raman Global University, Bhubaneswar  
**Department:** Artificial Intelligence & Data Science (AI&DS) / Computer Science and Engineering  
**Faculty Mentors:** **Prof. Priya Mishra** & **Ms. Pujarani Khuntia**  

---

### 👥 Student Team
| S.No. | Student Name | Registration No. | Project Contribution |
| :---: | :--- | :---: | :--- |
| 1 | **Rony Jagdev** | `2401020379` | ML Classification Pipeline & Model Training |
| 2 | **Suraj Kumar Mahto** | `2401020411` | Academic Data Warehouse Star Schema & ETL |
| 3 | **Anshu Kumar** | `2401020813` | Attendance Analytics & Feature Engineering |
| 4 | **Bhim Prasad Adhikari** | `2401021135` | Streamlit Risk Prediction Platform (`app.py`) |
| 5 | **Aayushman Baral** | `2401020764` | Empirical Analysis & Documentation |

---

### 🚀 Setup & Execution Guide for Teammates

#### 1. Clone the Repository
```bash
git clone https://github.com/BhimPrasadAdhikari/Student-Dropout-.git
cd Student-Dropout-
```

#### 2. Create and Activate Virtual Environment
```bash
# Windows
python -m venv .venv
.\.venv\Scripts\activate

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

#### 3. Install Required Dependencies
```bash
pip install -r requirements.txt
```

#### 4. Run the Machine Learning Pipeline (Optional / Pre-trained)
To re-train models, generate benchmark figures, and export artifacts:
```bash
python train_models.py
```

#### 5. Launch the Streamlit Prediction Console
```bash
streamlit run app.py
```
Or simply double-click `run_app.bat` on Windows.  
Open your browser at: `http://localhost:8501`

---

### 🤖 Machine Learning Model Benchmarks
Evaluated on 4,424 student records using **5-Fold Stratified Cross-Validation**:

| Model Name | Accuracy | Precision | Recall | F1-Score | ROC-AUC | 5-Fold CV Mean |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Support Vector Machine (RBF)** | **93.33%** | **89.75%** | **89.44%** | **0.8959** | **0.9785** | **0.9175** |
| **Gradient Boosting (GBDT)** | **93.11%** | **88.85%** | **89.79%** | **0.8932** | **0.9803** | **0.9181** |
| **Logistic Regression** | 93.11% | 89.12% | 89.44% | 0.8928 | **0.9804** | 0.9209 |
| **XGBoost Classifier** | 92.99% | 89.36% | 88.73% | 0.8905 | 0.9803 | 0.9209 |
| **Random Forest Classifier** | 92.66% | 87.63% | **89.79%** | 0.8870 | 0.9776 | 0.9161 |
| **K-Nearest Neighbors (KNN)** | 91.75% | 89.51% | 84.15% | 0.8675 | 0.9660 | 0.9011 |
| **Decision Tree (CART)** | 91.07% | 84.98% | 87.68% | 0.8631 | 0.9440 | 0.9014 |

---

### 📂 Repository File Structure
```
Student-Dropout-/
│
├── .streamlit/
│   └── config.toml          # Streamlit theme configuration
│
├── app.py                   # Streamlit Early-Warning & Risk Prediction Console
├── train_models.py          # Complete ML training & evaluation pipeline
├── run_app.bat              # One-click Windows launcher
├── requirements.txt         # Package dependencies
├── README.md                # Project documentation
│
├── data/
│   ├── data.csv             # Raw UCI/Kaggle dataset (4,424 records)
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

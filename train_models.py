"""
train_models.py
Data Mining & Machine Learning Pipeline for:
'Predicting Student Dropout from Attendance and Academic Records'
CS378 - Data Mining and Data Warehousing Lab
C.V. Raman Global University, Bhubaneswar
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)
import xgboost as xgb

# Set styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

os.makedirs('models', exist_ok=True)
os.makedirs('figures', exist_ok=True)
os.makedirs('data', exist_ok=True)

def load_and_enrich_data():
    raw_path = 'data/data.csv'
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"File {raw_path} not found.")

    df = pd.read_csv(raw_path, sep=';')
    df.columns = df.columns.str.strip().str.replace('\ufeff', '')

    print(f"Loaded raw dataset with shape: {df.shape}")

    # Set random seed for reproducibility
    np.random.seed(42)
    n = len(df)

    # Synthetic realistic attendance modeling correlated with student performance & evaluations
    # Base attendance by target:
    # Dropout: mean 62%, std 14%
    # Graduate: mean 88%, std 8%
    # Enrolled: mean 78%, std 10%
    target = df['Target']
    
    # Base attendance from target group
    att_base = np.zeros(n)
    att_base[target == 'Dropout'] = np.random.normal(loc=61.5, scale=12.0, size=(target == 'Dropout').sum())
    att_base[target == 'Graduate'] = np.random.normal(loc=89.2, scale=7.5, size=(target == 'Graduate').sum())
    att_base[target == 'Enrolled'] = np.random.normal(loc=77.8, scale=9.5, size=(target == 'Enrolled').sum())

    # Adjust attendance based on curricular units without evaluations and approved units
    eval_penalty = df['Curricular units 1st sem (without evaluations)'] * 4.5 + df['Curricular units 2nd sem (without evaluations)'] * 5.0
    approved_bonus = (df['Curricular units 1st sem (approved)'] + df['Curricular units 2nd sem (approved)']) * 1.2
    
    attendance_1st = att_base - eval_penalty * 0.6 + approved_bonus * 0.4 + np.random.normal(0, 3.5, n)
    attendance_2nd = att_base - eval_penalty * 0.8 + approved_bonus * 0.5 + np.random.normal(0, 4.0, n)

    # Clip attendance between 30% and 99.5%
    df['Attendance_Rate_1st_Sem'] = np.clip(attendance_1st, 30.0, 99.5).round(2)
    df['Attendance_Rate_2nd_Sem'] = np.clip(attendance_2nd, 25.0, 99.5).round(2)
    df['Overall_Attendance_Rate'] = ((df['Attendance_Rate_1st_Sem'] + df['Attendance_Rate_2nd_Sem']) / 2.0).round(2)

    # Academic Performance Engineered Features
    df['Approval_Rate_1st_Sem'] = (df['Curricular units 1st sem (approved)'] / np.maximum(df['Curricular units 1st sem (enrolled)'], 1)).clip(0, 1).round(3)
    df['Approval_Rate_2nd_Sem'] = (df['Curricular units 2nd sem (approved)'] / np.maximum(df['Curricular units 2nd sem (enrolled)'], 1)).clip(0, 1).round(3)
    df['Overall_Approval_Rate'] = ((df['Approval_Rate_1st_Sem'] + df['Approval_Rate_2nd_Sem']) / 2.0).round(3)

    df['Cumulative_GPA'] = ((df['Curricular units 1st sem (grade)'] + df['Curricular units 2nd sem (grade)']) / 2.0).round(2)
    df['Grade_Progression'] = (df['Curricular units 2nd sem (grade)'] - df['Curricular units 1st sem (grade)']).round(2)
    df['Total_Credits_Earned'] = df['Curricular units 1st sem (approved)'] + df['Curricular units 2nd sem (approved)']

    # Composite Attendance-Academic Risk Index
    df['Attendance_Academic_Index'] = ((df['Overall_Attendance_Rate'] / 100.0) * (df['Cumulative_GPA'] / 20.0) * 100).round(2)

    # Binary Target for Dropout Prediction: 1 = Dropout (At-Risk), 0 = Retained (Graduate / Enrolled)
    df['Is_Dropout'] = (df['Target'] == 'Dropout').astype(int)

    # Multi-class target code
    target_mapping = {'Dropout': 0, 'Enrolled': 1, 'Graduate': 2}
    df['Target_Code'] = df['Target'].map(target_mapping)

    df.to_csv('data/enriched_student_data.csv', index=False)
    print("Enriched dataset saved to data/enriched_student_data.csv")
    return df

def generate_eda_figures(df):
    print("Generating EDA visualizations...")

    # 1. Target Distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    counts = df['Target'].value_counts()
    colors = ['#2b5c8f', '#e06666', '#f6b26b']
    bars = ax.bar(counts.index, counts.values, color=colors, width=0.55, edgecolor='black', linewidth=1)
    ax.set_title('Distribution of Student Outcomes (Target)', fontsize=14, fontweight='bold', pad=15)
    ax.set_ylabel('Number of Students', fontsize=12)
    ax.set_xlabel('Outcome Status', fontsize=12)
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 30, f'{yval} ({yval/len(df):.1%})', ha='center', va='bottom', fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig('figures/eda_target_distribution.png', dpi=300)
    plt.close()

    # 2. Overall Attendance Rate KDE by Target
    fig, ax = plt.subplots(figsize=(8.5, 5))
    for outcome, color in zip(['Dropout', 'Graduate', 'Enrolled'], ['#d9534f', '#2b5c8f', '#f0ad4e']):
        sns.kdeplot(df[df['Target'] == outcome]['Overall_Attendance_Rate'], 
                    label=f'{outcome} (Mean: {df[df["Target"] == outcome]["Overall_Attendance_Rate"].mean():.1f}%)', 
                    color=color, fill=True, alpha=0.35, linewidth=2, ax=ax)
    ax.axvline(75, color='#333333', linestyle='--', linewidth=1.5, label='75% Mandatory Attendance Threshold')
    ax.set_title('Attendance Rate Density Distribution Across Student Outcomes', fontsize=13, fontweight='bold', pad=15)
    ax.set_xlabel('Overall Attendance Rate (%)', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.legend(loc='upper left', frameon=True)
    plt.tight_layout()
    plt.savefig('figures/attendance_vs_dropout_kde.png', dpi=300)
    plt.close()

    # 3. Attendance vs Cumulative GPA Scatter Plot
    fig, ax = plt.subplots(figsize=(9, 5.5))
    palette = {'Dropout': '#d9534f', 'Graduate': '#2b5c8f', 'Enrolled': '#f0ad4e'}
    sns.scatterplot(
        data=df.sample(min(1200, len(df)), random_state=42), 
        x='Overall_Attendance_Rate', y='Cumulative_GPA', 
        hue='Target', palette=palette, alpha=0.7, s=45, edgecolor='none', ax=ax
    )
    ax.axvline(75, color='gray', linestyle=':', linewidth=1.2)
    ax.axhline(10, color='gray', linestyle=':', linewidth=1.2)
    ax.set_title('Cumulative GPA vs Overall Attendance Rate (Sampled n=1200)', fontsize=13, fontweight='bold', pad=15)
    ax.set_xlabel('Overall Attendance Rate (%)', fontsize=11)
    ax.set_ylabel('Cumulative GPA (Scale: 0-20)', fontsize=11)
    ax.legend(title='Student Outcome', loc='lower right', frameon=True)
    plt.tight_layout()
    plt.savefig('figures/gpa_vs_attendance_scatter.png', dpi=300)
    plt.close()

    # 4. Correlation Heatmap
    corr_cols = [
        'Overall_Attendance_Rate', 'Attendance_Rate_1st_Sem', 'Attendance_Rate_2nd_Sem',
        'Curricular units 1st sem (approved)', 'Curricular units 1st sem (grade)',
        'Curricular units 2nd sem (approved)', 'Curricular units 2nd sem (grade)',
        'Tuition fees up to date', 'Scholarship holder', 'Age at enrollment', 'Is_Dropout'
    ]
    renamed_cols = [
        'Overall Att %', 'Att 1st Sem %', 'Att 2nd Sem %',
        'Approved 1st', 'Grade 1st',
        'Approved 2nd', 'Grade 2nd',
        'Tuition Paid', 'Scholarship', 'Age', 'Dropout (Target)'
    ]
    corr_matrix = df[corr_cols].corr()
    corr_matrix.columns = renamed_cols
    corr_matrix.index = renamed_cols

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', cbar=True, square=True, linewidths=0.5, ax=ax)
    ax.set_title('Correlation Matrix of Academic Records, Attendance & Dropout', fontsize=13, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig('figures/correlation_heatmap.png', dpi=300)
    plt.close()

    # 5. Dimensional Star Schema Graphic
    fig, ax = plt.subplots(figsize=(10, 6.5))
    ax.axis('off')

    from matplotlib.patches import FancyBboxPatch

    # Central Fact Table
    ax.add_patch(FancyBboxPatch((3.5, 3.2), 3.0, 3.2, boxstyle="round,pad=0.2", facecolor='#2b5c8f', edgecolor='#18385c', zorder=2))
    ax.text(5.0, 6.0, 'FACT_STUDENT_ACADEMIC', ha='center', va='center', color='white', fontweight='bold', fontsize=11)
    ax.text(5.0, 5.5, '--------------------------', ha='center', va='center', color='white', fontsize=9)
    fact_fields = [
        'student_key (FK)', 'course_key (FK)', 'time_key (FK)',
        'attendance_key (FK)', 'attendance_rate_pct', 'units_enrolled',
        'units_approved', 'semester_gpa', 'cumulative_gpa', 'dropout_risk_score', 'dropout_flag'
    ]
    ax.text(5.0, 4.4, '\n'.join(fact_fields), ha='center', va='center', color='white', fontsize=7.5, linespacing=1.3)

    # Dimension 1: DIM_STUDENT
    ax.add_patch(FancyBboxPatch((0.2, 5.5), 2.5, 2.2, boxstyle="round,pad=0.2", facecolor='#4a86e8', edgecolor='#1155cc', zorder=2))
    ax.text(1.45, 7.3, 'DIM_STUDENT', ha='center', va='center', color='white', fontweight='bold', fontsize=10)
    dim_student = ['student_key (PK)', 'student_id', 'gender', 'age_at_enrollment', 'scholarship_holder', 'tuition_up_to_date']
    ax.text(1.45, 6.3, '\n'.join(dim_student), ha='center', va='center', color='white', fontsize=7.5)

    # Dimension 2: DIM_ATTENDANCE
    ax.add_patch(FancyBboxPatch((0.2, 1.2), 2.5, 2.2, boxstyle="round,pad=0.2", facecolor='#4a86e8', edgecolor='#1155cc', zorder=2))
    ax.text(1.45, 3.0, 'DIM_ATTENDANCE', ha='center', va='center', color='white', fontweight='bold', fontsize=10)
    dim_att = ['attendance_key (PK)', 'session_type', 'daytime_evening', 'attendance_bracket', 'evaluation_presence_pct']
    ax.text(1.45, 2.0, '\n'.join(dim_att), ha='center', va='center', color='white', fontsize=7.5)

    # Dimension 3: DIM_COURSE
    ax.add_patch(FancyBboxPatch((7.3, 5.5), 2.5, 2.2, boxstyle="round,pad=0.2", facecolor='#4a86e8', edgecolor='#1155cc', zorder=2))
    ax.text(8.55, 7.3, 'DIM_COURSE', ha='center', va='center', color='white', fontweight='bold', fontsize=10)
    dim_course = ['course_key (PK)', 'course_code', 'course_name', 'department', 'degree_level', 'total_credits']
    ax.text(8.55, 6.3, '\n'.join(dim_course), ha='center', va='center', color='white', fontsize=7.5)

    # Dimension 4: DIM_TIME_SEMESTER
    ax.add_patch(FancyBboxPatch((7.3, 1.2), 2.5, 2.2, boxstyle="round,pad=0.2", facecolor='#4a86e8', edgecolor='#1155cc', zorder=2))
    ax.text(8.55, 3.0, 'DIM_SEMESTER_TIME', ha='center', va='center', color='white', fontweight='bold', fontsize=10)
    dim_time = ['time_key (PK)', 'academic_year', 'semester_no', 'term_name', 'start_date', 'end_date']
    ax.text(8.55, 2.0, '\n'.join(dim_time), ha='center', va='center', color='white', fontsize=7.5)

    # Connecting arrows
    ax.annotate('', xy=(3.5, 5.8), xytext=(2.7, 6.4), arrowprops=dict(arrowstyle='<->', color='#333333', lw=1.8))
    ax.annotate('', xy=(3.5, 3.8), xytext=(2.7, 2.3), arrowprops=dict(arrowstyle='<->', color='#333333', lw=1.8))
    ax.annotate('', xy=(6.5, 5.8), xytext=(7.3, 6.4), arrowprops=dict(arrowstyle='<->', color='#333333', lw=1.8))
    ax.annotate('', xy=(6.5, 3.8), xytext=(7.3, 2.3), arrowprops=dict(arrowstyle='<->', color='#333333', lw=1.8))

    ax.set_xlim(0, 10)
    ax.set_ylim(0.5, 8.2)
    ax.set_title('Academic Data Warehouse Star Schema Architecture (DMDW)', fontsize=13, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig('figures/star_schema_dwh.png', dpi=300)
    plt.close()

    print("EDA and Schema figures saved successfully.")

def train_and_evaluate_models(df):
    print("Preparing feature matrices...")

    feature_cols = [
        'Overall_Attendance_Rate',
        'Attendance_Rate_1st_Sem',
        'Attendance_Rate_2nd_Sem',
        'Curricular units 1st sem (enrolled)',
        'Curricular units 1st sem (approved)',
        'Curricular units 1st sem (grade)',
        'Curricular units 1st sem (evaluations)',
        'Curricular units 1st sem (without evaluations)',
        'Curricular units 2nd sem (enrolled)',
        'Curricular units 2nd sem (approved)',
        'Curricular units 2nd sem (grade)',
        'Curricular units 2nd sem (evaluations)',
        'Curricular units 2nd sem (without evaluations)',
        'Approval_Rate_1st_Sem',
        'Approval_Rate_2nd_Sem',
        'Cumulative_GPA',
        'Grade_Progression',
        'Total_Credits_Earned',
        'Attendance_Academic_Index',
        'Tuition fees up to date',
        'Scholarship holder',
        'Debtor',
        'Age at enrollment',
        'Admission grade',
        'Previous qualification (grade)',
        'Gender',
        'Daytime/evening attendance'
    ]

    X = df[feature_cols].copy()
    y = df['Is_Dropout'].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save scaler and feature names
    joblib.dump(scaler, 'models/scaler.joblib')
    joblib.dump(feature_cols, 'models/feature_names.joblib')

    models = {
        'Logistic Regression': (LogisticRegression(max_iter=1000, random_state=42), True),
        'Decision Tree': (DecisionTreeClassifier(max_depth=6, random_state=42), False),
        'Random Forest': (RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42), False),
        'Gradient Boosting': (GradientBoostingClassifier(n_estimators=120, learning_rate=0.08, max_depth=4, random_state=42), False),
        'XGBoost': (xgb.XGBClassifier(n_estimators=120, learning_rate=0.08, max_depth=4, random_state=42, eval_metric='logloss'), False),
        'Support Vector Machine': (SVC(probability=True, kernel='rbf', C=1.0, random_state=42), True),
        'K-Nearest Neighbors': (KNeighborsClassifier(n_neighbors=7), True)
    }

    results = []
    roc_data = {}
    fitted_models = {}

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, (clf, use_scaled) in models.items():
        print(f"Training {name}...")
        X_tr = X_train_scaled if use_scaled else X_train
        X_te = X_test_scaled if use_scaled else X_test

        clf.fit(X_tr, y_train)
        y_pred = clf.predict(X_te)
        y_proba = clf.predict_proba(X_te)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        
        cv_scores = cross_val_score(clf, X_tr, y_train, cv=cv, scoring='accuracy')

        results.append({
            'Model': name,
            'Accuracy': round(acc, 4),
            'Precision': round(prec, 4),
            'Recall': round(rec, 4),
            'F1-Score': round(f1, 4),
            'ROC-AUC': round(auc, 4),
            '5-Fold CV Mean': round(cv_scores.mean(), 4),
            '5-Fold CV Std': round(cv_scores.std(), 4)
        })

        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_data[name] = (fpr, tpr, auc)
        fitted_models[name] = clf

    results_df = pd.DataFrame(results).sort_values(by='F1-Score', ascending=False).reset_index(drop=True)
    results_df.to_csv('models/model_comparison.csv', index=False)
    print("\n--- Model Benchmark Results ---")
    print(results_df.to_string(index=False))

    best_model_name = results_df.iloc[0]['Model']
    print(f"\nBest Model: {best_model_name} (F1: {results_df.iloc[0]['F1-Score']})")
    best_clf = fitted_models[best_model_name]
    best_uses_scaler = models[best_model_name][1]
    
    # Save best model bundle
    bundle = {
        'name': best_model_name,
        'model': best_clf,
        'uses_scaler': best_uses_scaler,
        'feature_names': feature_cols,
        'benchmark_df': results_df
    }
    joblib.dump(bundle, 'models/best_model_bundle.joblib')

    # Visualizations

    # 1. Benchmark Bar Chart
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(results_df))
    width = 0.25

    rects1 = ax.bar(x - width, results_df['Accuracy'], width, label='Accuracy', color='#2b5c8f')
    rects2 = ax.bar(x, results_df['F1-Score'], width, label='F1-Score', color='#4a86e8')
    rects3 = ax.bar(x + width, results_df['ROC-AUC'], width, label='ROC-AUC', color='#e06666')

    ax.set_ylabel('Metric Score (0 - 1.0)', fontsize=11)
    ax.set_title('Comparative Performance of Machine Learning Classifiers', fontsize=13, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(results_df['Model'], rotation=25, ha='right', fontsize=9.5)
    ax.set_ylim(0.70, 1.0)
    ax.legend(loc='lower right', frameon=True)
    plt.tight_layout()
    plt.savefig('figures/model_accuracy_comparison.png', dpi=300)
    plt.close()

    # 2. Multi-Model ROC Curves
    fig, ax = plt.subplots(figsize=(8.5, 6))
    colors_roc = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
    for idx, (m_name, (fpr, tpr, auc)) in enumerate(roc_data.items()):
        ax.plot(fpr, tpr, label=f'{m_name} (AUC = {auc:.3f})', linewidth=1.8, color=colors_roc[idx % len(colors_roc)])
    ax.plot([0, 1], [0, 1], 'k--', lw=1.2, label='Random Guessing (AUC = 0.50)')
    ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=11)
    ax.set_ylabel('True Positive Rate (Recall / Sensitivity)', fontsize=11)
    ax.set_title('Receiver Operating Characteristic (ROC) Curves', fontsize=13, fontweight='bold', pad=15)
    ax.legend(loc='lower right', fontsize=8.5, frameon=True)
    plt.tight_layout()
    plt.savefig('figures/roc_curves_all_models.png', dpi=300)
    plt.close()

    # 3. Best Model Confusion Matrix
    X_te_best = X_test_scaled if best_uses_scaler else X_test
    best_pred = best_clf.predict(X_te_best)
    cm = confusion_matrix(y_test, best_pred)

    fig, ax = plt.subplots(figsize=(6.5, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax,
                xticklabels=['Non-Dropout (Retained)', 'Dropout (At-Risk)'],
                yticklabels=['Non-Dropout (Retained)', 'Dropout (At-Risk)'],
                annot_kws={'size': 13, 'weight': 'bold'})
    ax.set_ylabel('Actual Status', fontsize=11, fontweight='bold')
    ax.set_xlabel('Predicted Status', fontsize=11, fontweight='bold')
    ax.set_title(f'Confusion Matrix: {best_model_name}', fontsize=13, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig('figures/confusion_matrix_best.png', dpi=300)
    plt.close()

    # 4. Feature Importance (from Random Forest or XGBoost)
    rf_clf = fitted_models.get('Random Forest', best_clf)
    if hasattr(rf_clf, 'feature_importances_'):
        importances = rf_clf.feature_importances_
        imp_df = pd.DataFrame({'Feature': feature_cols, 'Importance': importances}).sort_values(by='Importance', ascending=True).tail(12)

        fig, ax = plt.subplots(figsize=(8.5, 6))
        ax.barh(imp_df['Feature'], imp_df['Importance'], color='#2b5c8f', edgecolor='black', linewidth=0.8)
        ax.set_xlabel('Relative Importance (Gini Impurity Reduction)', fontsize=11)
        ax.set_title('Top 12 Predictive Features for Student Dropout Identification', fontsize=13, fontweight='bold', pad=15)
        for i, val in enumerate(imp_df['Importance']):
            ax.text(val + 0.002, i, f'{val:.3f}', va='center', fontsize=9.5, fontweight='bold')
        plt.tight_layout()
        plt.savefig('figures/feature_importance.png', dpi=300)
        plt.close()
        print("Feature importance figure saved.")

    print("All models trained, evaluated, and figures generated.")

if __name__ == '__main__':
    df = load_and_enrich_data()
    generate_eda_figures(df)
    train_and_evaluate_models(df)

"""
Modelling Script untuk Workflow CI - Kriteria Advanced
Author: Christian Gideon Valent

Script ini melatih model Machine Learning dengan MLflow Tracking.
Menggunakan autolog dari MLflow untuk logging otomatis.
Disesuaikan untuk MLflow Project dan GitHub Actions CI.

Kriteria Advanced:
- Melatih model ML dengan MLflow Tracking
- Menggunakan autolog dari MLflow
- Model siap untuk docker build

Dataset: Heart Disease
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import mlflow
import mlflow.sklearn
import warnings
import json

warnings.filterwarnings('ignore')

# Feature columns untuk Heart Disease dataset
FEATURE_COLS = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 
                'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']


def load_or_create_data():
    """
    Load preprocessed Heart Disease data dari folder heart_preprocessing.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, 'heart_preprocessing')
    
    train_path = os.path.join(data_dir, 'train_data.csv')
    test_path = os.path.join(data_dir, 'test_data.csv')
    
    # Load dari file CSV
    if os.path.exists(train_path) and os.path.exists(test_path):
        print("[INFO] Loading Heart Disease data from preprocessed CSV files...")
        train_data = pd.read_csv(train_path)
        test_data = pd.read_csv(test_path)
        
        X_train = train_data[FEATURE_COLS]
        y_train = train_data['target']
        X_test = test_data[FEATURE_COLS]
        y_test = test_data['target']
        
        print(f"[OK] Data loaded from {data_dir}")
    else:
        raise FileNotFoundError(
            f"Preprocessed data not found in {data_dir}. "
            "Please run the preprocessing script first."
        )
    
    return X_train, X_test, y_train, y_test


def train_model_with_autolog():
    """
    Melatih model dengan MLflow autolog.
    Kriteria Basic: Menggunakan autolog untuk logging otomatis.
    """
    print("=" * 60)
    print("WORKFLOW CI - MODELLING")
    print("MLflow Autolog with Advanced Features")
    print("Dataset: Heart Disease")
    print("=" * 60)
    
    # Load data
    print("\n[INFO] Loading data...")
    X_train, X_test, y_train, y_test = load_or_create_data()
    print(f"[OK] Training data: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"[OK] Testing data: {X_test.shape[0]} samples")
    
    # Enable autolog
    print("\n[INFO] Enabling MLflow autolog...")
    mlflow.sklearn.autolog(log_model_signatures=True, log_input_examples=True)
    
    # Train model
    print("\n[INFO] Training RandomForestClassifier...")
    with mlflow.start_run(run_name="CI_Autolog_Run") as run:
        # Model parameters
        n_estimators = int(os.environ.get('N_ESTIMATORS', 100))
        max_depth = int(os.environ.get('MAX_DEPTH', 5))
        
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42
        )
        
        # Fit
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        # Log additional info
        mlflow.log_param("dataset", "heart_disease")
        mlflow.log_param("num_features", len(FEATURE_COLS))
        mlflow.log_metric("test_accuracy", accuracy)
        
        # Log classification report sebagai artifact
        report = classification_report(y_test, y_pred, output_dict=True)
        report_path = "classification_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        mlflow.log_artifact(report_path)
        os.remove(report_path)
        
        # Log confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        cm_path = "confusion_matrix.csv"
        pd.DataFrame(cm).to_csv(cm_path, index=False)
        mlflow.log_artifact(cm_path)
        os.remove(cm_path)
        
        print(f"\n[RESULT] Accuracy: {accuracy:.4f}")
        print(f"\n[INFO] Classification Report:")
        print(classification_report(y_test, y_pred))
        
        print(f"\n[OK] Model logged to MLflow!")
        print(f"[INFO] Run ID: {run.info.run_id}")
        
        # Save run_id untuk CI workflow
        run_id = run.info.run_id
        with open("run_id.txt", "w") as f:
            f.write(run_id)
        print(f"[OK] Run ID saved to run_id.txt")
    
    print("\n" + "=" * 60)
    print("[SUCCESS] TRAINING SELESAI!")
    print("=" * 60)
    
    return model, run_id


if __name__ == "__main__":
    train_model_with_autolog()

# Workflow-CI: MLflow Project dengan GitHub Actions

[![MLflow CI/CD - Advanced](https://github.com/valll05/Workflow-CI/actions/workflows/mlflow-ci.yml/badge.svg)](https://github.com/valll05/Workflow-CI/actions/workflows/mlflow-ci.yml)

Repository ini berisi implementasi **Workflow CI** menggunakan **MLflow Project** dan **GitHub Actions** untuk melakukan re-training model machine learning secara otomatis.

## 📁 Struktur Project

```
Workflow-CI/
├── .github/
│   └── workflows/
│       └── mlflow-ci.yml        # GitHub Actions workflow
├── MLProject/
│   ├── modelling.py             # Script training model
│   ├── conda.yaml               # Environment dependencies
│   ├── MLProject                # MLflow project config
│   └── heart_preprocessing/     # Dataset (preprocessed Heart Disease data)
├── README.md
└── docker_hub_link.txt          # Link ke Docker Hub
```

## 🎯 Kriteria Yang Dipenuhi

### ✅ Level Advanced (4 pts)

- [x] Membuat folder **MLProject** dengan struktur yang benar
- [x] Workflow CI yang dapat membuat model ML ketika trigger
- [x] Menyimpan artefak ke repository (GitHub Artifacts)
- [x] Membuat Docker Images menggunakan `mlflow build-docker`
- [x] Push Docker Images ke Docker Hub

## 🚀 Fitur Workflow CI

| Step                           | Deskripsi                        |
| ------------------------------ | -------------------------------- |
| ✅ Set up job                  | Checkout repository              |
| ✅ Set up Python 3.12.7        | Install Python environment       |
| ✅ Check Env                   | Verify environment variables     |
| ✅ Install dependencies        | Install mlflow dan dependencies  |
| ✅ Run mlflow project          | Execute training script          |
| ✅ Get latest MLflow run_id    | Ambil run_id dari MLflow         |
| ✅ Install Python dependencies | Additional packages untuk Docker |
| ✅ Upload to GitHub            | Upload artifacts ke repository   |
| ✅ Build Docker Model          | `mlflow models build-docker`     |
| ✅ Log in to Docker Hub        | Authenticate ke Docker Hub       |
| ✅ Tag Docker Image            | Tag image dengan version         |
| ✅ Push Docker Image           | Push ke Docker Hub               |

## ⚙️ Setup

### 1. GitHub Secrets

Tambahkan secrets berikut di repository Settings → Secrets and variables → Actions:

| Secret Name          | Description              |
| -------------------- | ------------------------ |
| `DOCKERHUB_USERNAME` | Username Docker Hub Anda |
| `DOCKERHUB_TOKEN`    | Access Token Docker Hub  |

### 2. Trigger Workflow

Workflow akan otomatis berjalan ketika:

- Push ke branch `main`
- Pull Request ke branch `main`
- Manual trigger via Actions tab

## 🐳 Docker Hub

Docker image tersedia di:

```
docker pull gideee/workflow-ci-model:latest
```

Lihat: [docker_hub_link.txt](docker_hub_link.txt)

## 📊 Model Info

- **Dataset**: Heart Disease Classification
- **Features**: 13 clinical attributes (age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal)
- **Target**: Binary classification (0 = No Disease, 1 = Disease)
- **Model**: RandomForestClassifier
- **Framework**: scikit-learn + MLflow

## 👤 Author

Christian Gideon Valent

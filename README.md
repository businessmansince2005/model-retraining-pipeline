# Model Retraining Pipeline (CI/CD for ML)

This project demonstrates a full **MLOps lifecycle**: models retrain automatically when new data is pushed, versioning is handled with MLflow, and the latest model is deployed via FastAPI + Docker.

## 🚀 Why it matters
Recruiters see that you understand:
- Model lifecycle management
- CI/CD automation
- Deployment pipelines
- Versioning and reproducibility

## 📐 Architecture
New Data → GitHub Push → CI/CD Trigger
→ Retrain Model (MLflow)
→ Register New Model Version
→ Build Docker Image
→ Deploy FastAPI with Latest Model

## 🛠 Tech Stack
- MLflow (model tracking & versioning)
- GitHub Actions (CI/CD automation)
- FastAPI (serving predictions)
- Docker (containerization)

## ⚡ How to run locally
`ash
pip install -r requirements.txt
python scripts/train.py
uvicorn app.app:app --reload
##🔮 API Usage
->Health check
curl http://localhost:8000/health
-Prediction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"feature1": 0.5, "feature2": -1.2}'
-response
{"prediction": 0.82}
✅ CI/CD
Workflow: .github/workflows/train.yml

Trigger: push new data/code

Action: retrains model, logs metrics, saves latest version
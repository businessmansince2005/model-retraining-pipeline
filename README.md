
# Model Retraining Pipeline (CI/CD for ML)

This project demonstrates a full **MLOps lifecycle**: models retrain automatically when new data is pushed, versioning is handled with MLflow, and the latest model is deployed via FastAPI + Docker.

## 🚀 Live Demo
Your service is live on Render:  
👉 [https://model-retraining-pipeline.onrender.com](https://model-retraining-pipeline.onrender.com)

- Health check: [https://model-retraining-pipeline.onrender.com/health](https://model-retraining-pipeline.onrender.com/health)  
- Swagger docs: [https://model-retraining-pipeline.onrender.com/docs](https://model-retraining-pipeline.onrender.com/docs)

## 📐 Architecture
```
New Data → GitHub Push → CI/CD Trigger
        → Retrain Model (MLflow)
        → Register New Model Version
        → Build Docker Image
        → Deploy FastAPI with Latest Model
```

## 🛠 Tech Stack
- MLflow (model tracking & versioning)
- GitHub Actions (CI/CD automation)
- FastAPI (serving predictions)
- Docker (containerization)
- Render (deployment)

## ✅ CI/CD Status
![CI](https://github.com/businessmansince2005/model-retraining-pipeline/actions/workflows/train.yml/badge.svg)

## ⚡ How to run locally

pip install -r requirements.txt
python scripts/train.py
uvicorn app.app:app --reload
```

## 🔮 API Usage
**Prediction**

curl -X POST "https://model-retraining-pipeline.onrender.com/predict" \
  -H "Content-Type: application/json" \
  -d '{"feature1": 0.5, "feature2": -0.3}'
```

Response:

{"prediction": 0.90}
```
```

---

⚡ Tactical checkpoint: once you paste this into `README.md`, commit and push, your repo will be fully recruiter‑ready with **live demo links** and **CI badge**.

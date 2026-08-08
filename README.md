# 🌿 Herbal-AI

> **AI-Powered Skin Disease Detection & Medicinal Herb Intelligence Platform**

Herbal-AI is an end-to-end AI healthcare platform that combines Computer Vision, Large Language Models (LLMs), Explainable AI, and MLOps to assist users in identifying skin diseases and medicinal plants while providing AI-generated medical summaries and herbal recommendations.

---

## ✨ Features

### 🩺 Skin Disease Detection
- Upload an image of a skin condition
- Deep Learning model predicts the disease
- Confidence score for predictions
- Supports 22 skin disease classes

### 🌿 Medicinal Herb Identification
- Identify medicinal plants from images
- AI-powered herbal recommendations
- Traditional medicinal information

### 🧠 AI Medical Assistant
- Llama 3.2 powered chatbot (Ollama)
- Disease explanation
- Herbal usage guidance
- Preventive care suggestions

### 📊 Explainable AI
- Grad-CAM heatmaps
- Visual explanation of model decisions
- Confidence visualization

### 📄 Medical Report Generation
- Downloadable PDF reports
- Prediction summary
- Herbal recommendations
- AI-generated explanations

### 🔐 Authentication
- JWT Authentication
- Secure API
- Cookie-based sessions

---

# 🏗️ Project Architecture

```text
                    ┌────────────────────────────┐
                    │       Next.js Frontend      │
                    └──────────────┬─────────────┘
                                   │
                          REST API (FastAPI)
                                   │
                ┌──────────────────┼──────────────────┐
                │                  │                  │
         Skin Disease AI      Herb AI          Llama 3.2
         (EfficientNetV2)  (EfficientNetV2)     (Ollama)
                │                  │                  │
                └──────────────────┼──────────────────┘
                                   │
                       Recommendation Engine
                                   │
                        PDF Report Generator
```

---

# 🛠️ Tech Stack

## Frontend

- Next.js
- React
- TypeScript
- Tailwind CSS
- Framer Motion
- shadcn/ui

## Backend

- FastAPI
- Python
- Pydantic
- JWT Authentication

## AI / ML

- PyTorch
- EfficientNetV2
- OpenCV
- Grad-CAM
- Ollama
- Llama 3.2

## MLOps (Upcoming)

- Git
- DVC
- MLflow
- Docker
- GitHub Actions
- Kubernetes
- AWS
- Prometheus
- Grafana

---

# 📂 Project Structure

```text
Herbal-AI
│
├── ai/
│   ├── image_classifier/
│   ├── herb/
│   ├── llm/
│   ├── preprocessing/
│   ├── recommendation/
│   └── explainability/
│
├── backend/
│
├── frontend/
│
├── tests/
│
├── requirements.txt
└── README.md
```

---

# 🚀 Installation

## Clone Repository

```bash
git clone https://github.com/Bareddycharitha/Herbal-AI.git

cd Herbal-AI
```

---

## Backend

```bash
conda create -n HerbalAI python=3.13

conda activate HerbalAI

pip install -r requirements.txt
```

---

## Frontend

```bash
cd frontend

npm install

uvicorn backend.app.main:app --reload
```

-----

## Backend

```bash
uvicorn backend.app.main:app --reload
```

---

# 🌐 API

| Endpoint | Description |
|----------|-------------|
| `/api/v1/predict` | Skin disease prediction |
| `/api/v1/herb` | Herb identification |
| `/api/v1/chat` | AI chatbot |
| `/api/v1/report` | Generate PDF report |
| `/health` | Health check |
| `/ready` | Readiness check |

---

# 📊 Model Information

## Skin Disease Model

- EfficientNetV2-S
- 22 Disease Classes
- PyTorch

## Herb Model

- EfficientNetV2-S
- Medicinal Plant Identification

---

# 🔍 Explainable AI

Herbal-AI uses **Grad-CAM** to visualize the important regions of an image that influence the model's predictions, improving transparency and trust.


# 📌 Future Improvements

- Multi-language support
- Voice Assistant
- Mobile Application
- OCR Prescription Reader
- Doctor Dashboard
- Patient History
- Multi-model Ensemble
- Cloud Deployment

---

# 👩‍💻 Author

**Bareddy Charitha**


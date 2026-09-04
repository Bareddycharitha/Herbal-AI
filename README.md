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

## 1. Backend setup (one-time)

```bash
# Create the virtual environment (already present in this repo as .venv/)
python -m venv .venv

# Activate it
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Windows (cmd):
.venv\Scripts\activate.bat
# macOS / Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# (Optional) Copy the env template and edit it
cp backend/.env.example .env
```

---

## 2. Frontend setup (one-time)

```bash
cd frontend
npm install
cp .env.example .env.local   # optional — defaults already point at localhost:8000
```

---

## ▶ Run the app

You need **two terminals** running side by side.

### Terminal 1 — Backend (FastAPI on :8000)

```bash
# from the repo root, with .venv activated
uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
```

### Terminal 2 — Frontend (Next.js on :3000)

```bash
cd frontend
npm run dev
```

Then open <http://localhost:3000> in your browser. The Next.js dev server
will proxy API calls to the FastAPI backend on `localhost:8000`.

### Verify it works

- `curl http://localhost:8000/health` → `{"status":"healthy",...}`
- `curl -I http://localhost:8000/ready` → `200 OK` when all checkpoints are
  present, `503` with a per-check `file_exists: false` field when not.
  The response must include `Access-Control-Allow-Origin: http://localhost:3000`
  so the browser doesn't show a misleading "Network Error" toast.
- Open <http://localhost:3000/diagnose> and pick a workflow to test the
  full upload → prediction → result flow.

> **Heads up — model checkpoints**: the repository does not include
> trained model weights. Until a `best_model.pth` is placed at
> `ai/checkpoints/best_model.pth` (skin disease) and
> `ai/herb/checkpoints/best_model.pth` (herbs), the corresponding
> endpoints return `503 MODEL_LOAD_ERROR` with a clear message — **not**
> a generic 500 / "Network Error".

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


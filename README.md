🌿 Herbal-AI

AI-Powered Skin Disease Detection, Medicinal Herb Identification & Intelligent Healthcare Assistant

Herbal-AI is an end-to-end AI-powered healthcare and herbal intelligence platform that combines Computer Vision, Deep Learning, Explainable AI, Large Language Models, Retrieval/Knowledge-Based Recommendations, Authentication, MLOps, and Cloud Deployment into a single application.

The platform allows users to upload images and receive AI-assisted analysis for skin conditions and medicinal plants, along with confidence scores, explainability through Grad-CAM, disease/herbal information, AI-generated summaries, recommendations, conversational assistance, and downloadable PDF reports.

⚠️ Medical Disclaimer: Herbal-AI is an educational and research-oriented AI system. Its predictions and generated information are not a medical diagnosis and should not replace evaluation, diagnosis, or treatment from a qualified healthcare professional.

✨ Key Features

🩺 Skin Disease Detection
Upload a skin image through the web interface.
Universal image classifier first determines whether the image belongs to:
Skin
Medicinal plant
Other
Skin images are routed to the dedicated skin disease classifier.
Supports 22 skin disease/condition classes.
Returns:
Predicted condition
Confidence score
Confidence level
Top predictions
Disease information
Recommended herbs where applicable
Includes image validation and out-of-distribution detection.

🌿 Medicinal Herb Identification

Herbal-AI includes a dedicated medicinal plant classifier based on EfficientNetV2-S.

The system can:

Identify medicinal plants from images.
Return the predicted herb and confidence.
Provide medicinal information.
Retrieve relevant information from the herbal knowledge base.
Provide herbal recommendations where applicable.

The trained herb model contains 40 classes in the production model.

🔀 Universal Image Classification

Before specialized analysis, Herbal-AI uses a universal image classifier to determine the type of uploaded image.

                    Uploaded Image
                          │
                          ▼
                Universal Classifier
                          │
             ┌────────────┼────────────┐
             ▼            ▼            ▼
           Skin       Medicinal       Other
             │            │
             ▼            ▼
       Skin Model      Herb Model

This routing layer helps ensure that images are sent to the appropriate specialized model instead of directly applying a specialized classifier to every input

The universal classifier uses EfficientNetV2-S and produces three classes:

0 → Skin
1 → Medicinal
2 → Other

🧠 AI Architecture
                         ┌──────────────────────┐
                         │     Next.js Web App   │
                         │ React + TypeScript    │
                         │ Tailwind + shadcn/ui  │
                         └───────────┬──────────┘
                                     │
                                  REST API
                                     │
                                     ▼
                         ┌──────────────────────┐
                         │      FastAPI API      │
                         │ Authentication        │
                         │ Prediction Routing    │
                         │ History               │
                         │ Reports               │
                         └───────────┬──────────┘
                                     │
                                     ▼
                       ┌──────────────────────────┐
                       │ Universal Image Classifier│
                       │      EfficientNetV2-S     │
                       └────────────┬─────────────┘
                                    │
                     ┌──────────────┼──────────────┐
                     │              │              │
                     ▼              ▼              ▼
                   Skin         Medicinal        Other
                     │              │
                     ▼              ▼
             ┌─────────────┐ ┌─────────────┐
             │ Skin Model  │ │ Herb Model  │
             │ EfficientNet│ │ EfficientNet│
             │ V2-S        │ │ V2-S        │
             └──────┬──────┘ └──────┬──────┘
                    │               │
                    └───────┬───────┘
                            ▼
                ┌─────────────────────────┐
                │ Knowledge / Recommendation│
                │ Engine                    │
                └────────────┬────────────┘
                             │
             ┌───────────────┼────────────────┐
             ▼               ▼                ▼
        Disease Info     Herb Info       AI Summary
                                             │
                                             ▼
                                     OpenRouter LLM
                                             │
                                             ▼
                                        AI Chatbot
                                             │
                                             ▼
                                      PDF Report
🤖 AI / ML Pipeline

The complete prediction pipeline is:

User Upload
     │
     ▼
Image Validation
     │
     ▼
Universal Classification
     │
     ├──────── Skin ────────► Skin Disease Model
     │
     ├──────── Medicinal ───► Herb Model
     │
     └──────── Other ───────► Rejected / Unsupported
                                  │
                                  ▼
                          Prediction + Confidence
                                  │
                    ┌─────────────┼─────────────┐
                    ▼             ▼             ▼
                  OOD         Grad-CAM      Knowledge Base
                    │             │             │
                    └─────────────┼─────────────┘
                                  ▼
                         User-facing Results
                                  │
                     ┌────────────┼────────────┐
                     ▼            ▼            ▼
                AI Summary      AI Chat      PDF Report
🔬 Machine Learning Models

Universal Classifier

Architecture: EfficientNetV2-S

Classes:

Index	Class
0	Skin
1	Medicinal
2	Other

The model is responsible for routing images to the appropriate specialized pipeline.

Skin Disease Classifier

Architecture: EfficientNetV2-S

Framework: PyTorch

Number of classes: 22

The production classifier supports the following categories:

Acne
Actinic_Keratosis
Benign_tumors
Bullous
Candidiasis
DrugEruption
Eczema
Infestations_Bites
Lichen
Lupus
Moles
Psoriasis
Rosacea
Seborrh_Keratoses
SkinCancer
Sun_Sunlight_Damage
Tinea
Unknown_Normal
Vascular_Tumors
Vasculitis
Vitiligo
Warts
Medicinal Herb Classifier

Architecture: EfficientNetV2-S

Framework: PyTorch

Number of production classes: 40

The model was trained specifically for medicinal plant identification.

The final test evaluation achieved:

Metric	Result
Accuracy	99.84%
Precision	99.85%
Recall	99.84%
F1 Score	99.84%

Test set:

Training images: 4,729
Validation images: 573
Test images: 643
Classes: 40
🧪 Out-of-Distribution Detection

Herbal-AI does not rely only on classification confidence.

The system calculates multiple OOD-related signals:

Energy score
Maximum Softmax Probability (MSP)
Prediction entropy
Combined OOD score

These signals are used to determine whether an input is potentially outside the expected distribution.

This helps reduce the risk of blindly presenting a specialized prediction for an inappropriate image.

🔍 Explainable AI — Grad-CAM

Herbal-AI uses Grad-CAM to provide visual explanations of model predictions.

Instead of displaying only:

Prediction: Eczema
Confidence: 80.09%

the application can also display an attention/heatmap visualization showing the image regions that contributed to the model's prediction.

This improves interpretability and makes the model's decision-making easier to inspect.

🌿 Knowledge & Recommendation Engine

The platform maintains structured knowledge bases for diseases and medicinal herbs.

The recommendation system connects model predictions with relevant knowledge.

For example:

Prediction
    │
    ▼
Disease → Knowledge Base
    │
    ├── Description
    ├── Symptoms
    ├── Causes
    ├── Prevention
    └── Related Herbs

The herbal knowledge base contains structured information including:

Common name
Botanical name
Benefits
Traditional uses
Preparation methods
Side effects
Contraindications
Evidence level
Other relevant information

The system also contains disease-to-herb mappings.

🤖 AI Medical Summary

After prediction, Herbal-AI can generate an AI-assisted summary using an LLM through OpenRouter.

The summary can explain:

What the model detected
What the confidence score means
General information about the condition
General precautions
Relevant herbal information

The application also provides a deterministic fallback response if the external LLM service is unavailable.

AI-generated summaries are informational and should not be interpreted as medical diagnosis or treatment instructions.

💬 AI Healthcare Chatbot

The application includes an AI conversational assistant.

Users can ask questions such as:

Explain this disease
What precautions should I take?
Are the recommended herbs safe?
How long does recovery usually take?
Can this condition spread?

The chatbot receives the prediction context and relevant disease/herbal information before generating its response.

The LLM layer currently uses OpenRouter, rather than a locally hosted Ollama deployment.

📄 PDF Medical Report

Users can generate a downloadable PDF report containing information such as:

Uploaded image
AI prediction
Confidence
Disease information
Herbal recommendations
AI-generated summary
Explainability information where available

This provides a portable record of the AI analysis.

🔐 Authentication
Herbal-AI uses Clerk for user authentication.

The backend verifies Clerk authentication tokens before processing protected user requests.

Authenticated functionality includes:

User profile
Prediction history
AI summary persistence
Protected API operations

🗄️ Database & User History

Supabase is used for application data management.

The platform stores information related to:

User profiles
Prediction history
AI-generated summaries
Associated prediction information

This allows users to access previous analyses instead of losing them after a session.

🛠️ Technology Stack
Frontend
Next.js
React
TypeScript
Tailwind CSS
shadcn/ui
Framer Motion
Axios
Backend
Python
FastAPI
Pydantic
Uvicorn
AI / Machine Learning
PyTorch
timm
EfficientNetV2-S
OpenCV
Grad-CAM
Custom OOD detection
OpenRouter
Large Language Models
Authentication & Data
Clerk
Supabase
MLOps
Git
DVC
MLflow
MLflow Model Registry
DagsHub
Docker
GitHub Actions
Cloud / Deployment
AWS EC2
AWS ECR
Amazon S3
Docker Compose

📊 MLOps Architecture

Herbal-AI includes an MLOps workflow for managing trained models and reproducibility.

Dataset
   │
   ▼
DVC
   │
   ▼
Training
   │
   ▼
Evaluation
   │
   ▼
MLflow Tracking
   │
   ▼
Model Artifacts
   │
   ▼
MLflow Model Registry
   │
   ▼
Model Validation
   │
   ▼
Docker
   │
   ▼
GitHub Actions
   │
   ▼
Amazon ECR
   │
   ▼
AWS EC2

📦 DVC

Dataset versioning is handled using DVC.

The project uses an S3-backed DVC remote to manage large datasets separately from Git.

This prevents large training datasets from being stored directly inside the Git repository.

📈 MLflow

MLflow is used for experiment tracking and model lifecycle management.

Tracked information includes:

Training parameters
Epoch metrics
Validation metrics
Test metrics
Model artifacts
Evaluation artifacts
Model metadata

The project also uses MLflow Model Registry.

The following production model families have been registered:

UniversalClassifier
HerbClassifier
SkinDiseaseClassifier

Each registered model can be retrieved and validated independently before deployment.

🐳 Docker

The application is containerized for reproducible deployment.

Production architecture:

Docker Compose
      │
      ├── Next.js Frontend
      │
      └── FastAPI Backend
              │
              ├── Universal Classifier
              ├── Skin Classifier
              ├── Herb Classifier
              └── OpenRouter

The production backend uses a CPU-optimized Python image to reduce deployment size and does not require a GPU on the production EC2 instance.

🔄 CI/CD

GitHub Actions is used as part of the deployment workflow.

The intended deployment flow is:

Developer Push
      │
      ▼
GitHub
      │
      ▼
GitHub Actions
      │
      ├── Tests
      ├── Validation
      ├── Docker Build
      └── ECR Push
              │
              ▼
          AWS Deployment

GitHub Actions is used to automate testing, Docker image building, and container image publishing as part of the deployment workflow.

☁️ AWS Deployment

The production application is deployed using:

AWS EC2
   │
   ├── Frontend Container
   │
   └── Backend Container
          │
          └── Production AI Models

Docker images are stored in Amazon ECR and deployed to an EC2 instance.

The backend exposes the FastAPI service while the Next.js frontend provides the user-facing application.

📁 Project Structure
Herbal-AI/
│
├── ai/
│   ├── image_classifier/
│   │   ├── config.py
│   │   ├── dataset.py
│   │   ├── evaluate.py
│   │   ├── inference.py
│   │   ├── model.py
│   │   ├── trainer.py
│   │   └── checkpoints/
│   │
│   ├── herb/
│   │   ├── config.py
│   │   ├── dataset.py
│   │   ├── inference.py
│   │   ├── knowledge_base.py
│   │   ├── model.py
│   │   ├── training/
│   │   ├── scripts/
│   │   └── checkpoints/
│   │
│   ├── llm/
│   ├── preprocessing/
│   ├── recommendation/
│   ├── explainability/
│   └── datasets/
│
├── backend/
│   ├── app/
│   │   ├── api/
│   │   ├── services/
│   │   ├── repositories/
│   │   ├── middleware.py
│   │   └── main.py
│   │
│   ├── Dockerfile
│   └── Dockerfile.prod
│
├── frontend/
│   ├── app/
│   ├── components/
│   ├── lib/
│   ├── providers/
│   └── types/
│
├── tests/
│
├── .github/
│   └── workflows/
│
├── docker-compose.yml
├── requirements.txt
├── requirements-prod.txt
└── README.md

🚀 Local Installation
Prerequisites

Recommended:

Python 3.11+
Node.js 20+
npm
Git
Git LFS/DVC where required
Access to the required environment variables
Trained model checkpoints
Clone Repository
git clone https://github.com/Bareddycharitha/Herbal-AI.git
cd Herbal-AI

🐍 Backend Setup

Create a virtual environment:

python -m venv .venv
Windows PowerShell
.venv\Scripts\Activate.ps1
Windows CMD
.venv\Scripts\activate.bat
Linux/macOS
source .venv/bin/activate

Install dependencies:

pip install -r requirements.txt

Configure the required environment variables using the project's environment template.

💻 Frontend Setup
cd frontend
npm install

Configure the frontend environment variables using:

frontend/.env.example

Then return to the repository root when necessary.

▶️ Run Locally
Terminal 1 — Backend

From the repository root:
uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000

Backend:
http://localhost:8000

API documentation:

http://localhost:8000/docs

Terminal 2 — Frontend
cd frontend
npm run dev

Frontend:
http://localhost:3000

🐳 Run With Docker

Build and start the application:

docker compose up --build

Check running containers:

docker compose ps

Stop:

docker compose down

🔌 API Endpoints
Endpoint	Method	Purpose
/health	GET	Application health check
/ready	GET	Model/application readiness
/api/v1/predict/	POST	Image prediction
/api/v1/gradcam/{prediction_id}	GET	Grad-CAM status/result
/api/v1/summary/	POST	AI-generated summary
/api/v1/chat/	POST	AI conversational assistant
/api/v1/history/	GET	User prediction history
/api/v1/report/	POST	PDF report generation
/api/v1/auth/me	GET	Current authenticated user

🧪 Testing & Verification

The project includes validation across multiple layers.

Model verification

Verified:

Universal Classifier → (1, 3)
Herb Classifier      → (1, 40)
Skin Classifier      → (1, 22)

The models were also verified through the MLflow Model Registry.

Production verification

The deployed backend has been verified for:

Health checks
Authentication
Image prediction
Universal routing
Skin disease inference
Herb inference
Grad-CAM
AI summaries
AI chatbot
Prediction history
CORS
Docker health checks

📊 Example Production Prediction

Example skin prediction:

Prediction: Eczema
Confidence: 80.09%

Top Predictions:
1. Eczema                  80.09%
2. Sun_Sunlight_Damage      4.21%
3. Lupus                    3.62%

Image Type: Skin
Classifier Confidence: 100%
OOD: False

🧠 Model Registry

Registered model families:

UniversalClassifier
HerbClassifier
SkinDiseaseClassifier

Each model is packaged and validated before being considered for production use.

This provides separation between:

Training
   ↓
Model Artifact
   ↓
Registry
   ↓
Validation
   ↓
Deployment

# 🔒 Security Considerations

The project separates sensitive configuration from source code.

Sensitive credentials should be supplied through environment variables and must never be committed to Git.

Examples include:

Clerk secrets
Supabase credentials
OpenRouter API keys
AWS credentials
MLflow credentials

Before publishing or sharing the repository, ensure:

.env files are excluded
API keys are rotated if accidentally exposed
Cloud credentials are not committed
Production secrets are not included in Docker images unnecessarily

# 🏆 What This Project Demonstrates

### Machine Learning
- Deep learning image classification
- Transfer learning with EfficientNetV2-S
- Multi-class classification
- Model evaluation
- Confidence estimation
- Out-of-distribution detection
- Explainable AI with Grad-CAM

### Generative AI
- LLM integration through OpenRouter
- Context-aware AI summaries
- Conversational AI
- Fallback handling for external LLM failures

### MLOps
- Dataset versioning with DVC
- Experiment tracking with MLflow
- MLflow Model Registry
- Model validation
- Dockerized inference
- CI/CD with GitHub Actions

### Cloud & Backend
- FastAPI REST APIs
- Clerk authentication
- Supabase persistence
- AWS EC2
- Amazon ECR
- S3-backed DVC storage

### Frontend
- Next.js
- TypeScript
- Tailwind CSS
- shadcn/ui
- Responsive AI application interface

⚠️ Limitations

Herbal-AI is a research and educational project and has several limitations:

Image-based classification cannot replace clinical examination.
Model performance depends on image quality and dataset distribution.
Predictions outside the training distribution may be unreliable.
AI-generated summaries can contain errors.
Herbal recommendations should not replace professional medical advice.
External LLM availability depends on the OpenRouter service and selected providers.
Production deployment currently uses a CPU-based backend, which trades inference speed for lower infrastructure cost.

🚀 Future Improvements

Potential future improvements include:

Multilingual healthcare assistance
Voice-based interaction
Mobile application
Prescription OCR
Doctor/clinician dashboard
More extensive patient history
Larger and more diverse datasets
Model ensemble improvements
Improved calibration
Advanced OOD detection
Clinical validation
More comprehensive monitoring
Kubernetes-based horizontal scaling
Prometheus/Grafana observability

These are future directions, not claims about the current production deployment.

Project Highlights

Herbal-AI demonstrates an end-to-end workflow covering:

Computer Vision
       +
Deep Learning
       +
Explainable AI
       +
LLMs
       +
Knowledge-Based Recommendations
       +
Authentication
       +
Database
       +
MLOps
       +
Model Registry
       +
Docker
       +
CI/CD
       +
AWS Deployment

The project was designed not only as an AI model but as a complete production-oriented AI application, connecting model development, evaluation, serving, frontend interaction, user authentication, data persistence, containerization, and cloud deployment.

👩‍💻 Author
Bareddy Charitha
B.Tech — Computer Science & Engineering (AI/ML)

GitHub:
https://github.com/Bareddycharitha

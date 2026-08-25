# Herbal-AI Project Report

## Project Overview

**Herbal-AI** is a production-ready AI-powered system for skin disease diagnosis and medicinal herb identification. It combines deep learning models with a knowledge base and LLM-powered chatbot to provide educational guidance on skin conditions and herbal remedies.

### Architecture Summary

```
┌─────────────────────────────────────────────────────────────────────┐
│                        HERBAL-AI SYSTEM                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐          │
│  │   Frontend   │◄──►│   Backend    │◄──►│   AI Models  │          │
│  │  (Next.js)   │    │  (FastAPI)   │    │  (PyTorch)   │          │
│  └──────────────┘    └──────────────┘    └──────────────┘          │
│       │                    │                    │                   │
│       ▼                    ▼                    ▼                   │
│  • Upload UI          • Universal Classifier  • EfficientNetV2-S   │
│  • Results Display    • Skin Disease Pipeline │ Skin Disease (22)  │
│  • Herb Identification│ • Herb Pipeline       │ Herb ID (classes)  │
│  • AI Chat            • PDF Reports           • Universal (3 cls)  │
│  • Theme Support      • Ollama LLM            • Grad-CAM           │
│  • Auth (JWT)         • JWT Auth              │                    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Tech Stack

### Frontend
| Technology | Version | Purpose |
|------------|---------|---------|
| Next.js | 16.2.10 | React framework with App Router |
| React | 19.2.4 | UI library |
| TypeScript | 5 | Type safety |
| Tailwind CSS | 4 | Styling |
| shadcn/ui + Radix | Latest | Accessible UI components |
| Framer Motion | 12.42.2 | Animations |
| Axios | 1.18.1 | HTTP client |
| next-themes | 0.4.6 | Dark/light mode |
| **ESLint** | ^9 | Code quality & error detection |
| **Prettier** | ^3.0.0 | Code formatting |

### Backend
| Technology | Version | Purpose |
|------------|---------|---------|
| FastAPI | 0.115.0 | Modern Python web framework |
| Uvicorn | 0.32.0 | ASGI server |
| Pydantic | 2.9.2 | Data validation |
| Pydantic Settings | 2.5.2 | Configuration management |
| Python | 3.11+ | Runtime |

### AI/ML
| Technology | Version | Purpose |
|------------|---------|---------|
| PyTorch | 2.5.1 | Deep learning framework |
| torchvision | 0.20.1 | Computer vision utilities |
| timm | 1.0.12 | Pre-trained models (EfficientNetV2-S) |
| Albumentations | 2.0.4 | Image augmentations |
| OpenCV | 4.10.0 | Image processing |
| Grad-CAM | 1.4.7 | Explainability |
| scikit-learn | 1.5.2 | Metrics, class weights |

### Authentication & Security
| Technology | Version | Purpose |
|------------|---------|---------|
| Argon2 (passlib) | Latest | Password hashing |
| python-jose | Latest | JWT tokens |
| OAuth2 | Built-in | Login flow |

### Observability & Testing
| Technology | Version | Purpose |
|------------|---------|---------|
| structlog | 24.1.0 | Structured JSON logging |
| Prometheus Client | 0.20.0 | Metrics |
| pytest | 8.3.4 | Testing |
| pytest-asyncio | 0.24.0 | Async testing |
| **ESLint** | ^9 | Code quality & error detection |
| **Prettier** | ^3.0.0 | Code formatting |

### Infrastructure
| Technology | Purpose |
|------------|---------|
| Ollama | Local LLM (llama3.2:3b) |
| Git | Version control |
| Docker | Containerization (Dockerfile) |

---

## Project Structure

```
Herbal-AI/
├── .env.example                 # Environment template
├── .gitignore                   # Git ignore rules
├── .eslintignore                # ESLint ignore rules
├── .eslintrc.js                 # ESLint configuration
├── .prettierignore              # Prettier ignore rules
├── .prettierrc                  # Prettier configuration
├── pytest.ini                   # Pytest configuration
├── requirements.txt             # Root Python dependencies
├── README.md                    # Project overview
├── report.md                    # This file
├── ai/                          # AI/ML modules
│   ├── config.py                # Global AI configuration
│   ├── __init__.py
│   ├── datasets/                # Knowledge bases
│   │   └── knowledge_base/
│   │       ├── disease_knowledge_base.json    # 22 skin diseases
│   │       └── herbal_knowledge_base.json     # Herb details
│   ├── explainability/
│   │   ├── gradcam_engine.py                  # Grad-CAM implementation
│   │   └── __init__.py
│   ├── herb/                     # Herb Identification Module
│   │   ├── config.py              # Herb model config
│   │   ├── model.py               # EfficientNetV2-S classifier
│   │   ├── dataset.py             # Data loaders
│   │   ├── inference.py           # Inference with OOD, calibration
│   │   ├── knowledge_base.py      # Herb KB with caching
│   │   ├── leaf_detector.py       # Binary leaf detection
│   │   ├── transforms.py          # Image transforms
│   │   ├── checkpoints/           # Model weights
│   │   ├── results/               # Training results
│   │   ├── scripts/               # Utility scripts
│   │   └── training/              # Training pipeline
│   ├── image_classifier/         # Universal 3-class Classifier
│   │   ├── config.py
│   │   ├── model.py
│   │   ├── dataset.py
│   │   ├── trainer.py
│   │   ├── evaluate.py
│   │   ├── inference.py           # Ensemble, OOD, calibration
│   │   ├── transforms.py
│   │   ├── utils.py
│   │   └── checkpoints/
│   ├── llm/                      # LLM Integration (Ollama)
│   │   ├── chatbot_engine.py      # Context-aware chatbot
│   │   ├── ollama_client.py       # Async client with resilience
│   │   ├── prompt_builder.py      # Prompt templates
│   │   ├── summary_engine.py      # AI summaries
│   │   └── __init__.py
│   ├── models/
│   │   └── efficientnet.py        # Shared model builder
│   ├── preprocessing/
│   │   ├── dataset.py             # SkinDisease dataset
│   │   ├── transforms.py          # Albumentations
│   │   └── __init__.py
│   ├── recommendation/           # Recommendation Pipelines
│   │   ├── recommendation_engine.py        # Skin disease pipeline
│   │   ├── herb_recommendation_engine.py   # Herb pipeline
│   │   ├── knowledge_base.py               # Disease KB (cached)
│   │   ├── herbal_knowledge_base.py        # Herb KB (cached)
│   │   └── __init__.py
│   ├── scripts/                   # Analysis scripts
│   ├── training/                  # Skin Disease Training
│   │   ├── trainer.py
│   │   ├── train.py
│   │   ├── inference.py
│   │   ├── callbacks.py
│   │   ├── loss.py                # Focal Loss
│   │   ├── ood_detection.py       # Energy/MSP/Entropy OOD
│   │   ├── calibration.py         # Temperature scaling
│   │   └── __init__.py
│   ├── utils/                     # Shared utilities
│   ├── validation/                # Image validators
│   └── __init__.py
├── backend/                       # FastAPI Backend
│   ├── requirements.txt           # Backend dependencies
│   ├── Dockerfile                 # Container definition
│   └── app/
│       ├── main.py                # FastAPI app + lifespan
│       ├── config.py              # Pydantic Settings config
│       ├── exceptions.py          # Custom exception hierarchy
│       ├── exception_handlers.py  # Global error handlers
│       ├── middleware.py          # Request ID, logging, rate limit
│       ├── dependencies.py        # Auth dependencies (RBAC)
│       ├── models/
│       │   └── user.py            # User models, in-memory store
│       ├── utils/
│       │   ├── auth.py            # JWT, password hashing
│       │   ├── file_validator.py  # Secure file upload validation
│       │   ├── logging.py         # Structlog setup
│       │   └── __init__.py
│       ├── schemas/
│       │   ├── request.py         # Request models
│       │   ├── response.py        # Response models
│       │   ├── report.py          # Report schema
│       │   └── __init__.py
│       ├── api/
│       │   ├── auth.py            # JWT auth endpoints
│       │   ├── prediction.py      # Universal /predict/
│       │   ├── herb.py            # /herb/ endpoint
│       │   ├── summary.py         # /summary/ endpoint
│       │   ├── chat.py            # /chat/ endpoint
│       │   ├── report.py          # /report/ PDF generation
│       │   └── __init__.py
│       ├── services/
│       │   ├── universal_classifier.py  # DI-compatible classifier
│       │   ├── pdf/                       # PDF report generation
│       │   │   ├── report_generator.py
│       │   │   ├── sections.py
│       │   │   ├── styles.py
│       │   │   └── assets/logo.png
│       │   └── __init__.py
│       └── __init__.py
├── frontend/                      # Next.js Frontend
│   ├── package.json               # Dependencies
│   ├── tsconfig.json              # TypeScript config
│   ├── next.config.ts             # Next.js config
│   ├── eslint.config.mjs          # ESLint config
│   ├── postcss.config.mjs         # PostCSS config
│   ├── .gitignore
│   ├── README.md
│   ├── CLAUDE.md
│   ├── AGENTS.md
│   ├── next-env.d.ts
│   ├── public/                    # Static assets
│   ├── app/                       # App Router pages
│   │   ├── layout.tsx             # Root layout + ThemeProvider
│   │   ├── page.tsx               # Home page
│   │   ├── globals.css            # Global styles
│   │   ├── diagnose/page.tsx      # Service selection
│   │   ├── skin-analysis/page.tsx # Skin upload
│   │   ├── herb-identification/page.tsx # Herb upload
│   │   ├── results/page.tsx       # Skin results
│   │   ├── herb-results/page.tsx  # Herb results
│   │   ├── chat/page.tsx          # AI Chat
│   │   └── about/page.tsx         # About page
│   ├── components/
│   │   ├── Navbar.tsx
│   │   ├── Hero.tsx
│   │   ├── Features.tsx
│   │   ├── Footer.tsx
│   │   ├── HowItWorks.tsx
│   │   ├── UploadCard.tsx         # Drag-drop upload
│   │   ├── ResultCard.tsx
│   │   ├── ThemeToggle.tsx
│   │   ├── ui/                    # shadcn/ui components
│   ├── lib/
│   │   ├── api.ts                 # Axios API client
│   │   └── utils.ts
│   ├── providers/
│   │   └── ThemeProvider.tsx
│   └── types/
│       └── index.ts               # TypeScript types
└── tests/                         # Test Suite (149 tests)
    ├── conftest.py                # Shared fixtures
    ├── test_auth.py               # Auth tests (31 tests)
    ├── test_config.py             # Config tests
    ├── test_exceptions.py         # Exception tests
    ├── test_file_validator.py     # File validation tests
    ├── test_image_validator.py    # Image validation tests
    ├── test_knowledge_base.py     # KB tests
    ├── test_ollama_client.py      # Ollama client tests
    ├── test_api_endpoints.py      # API integration tests
    └── test_*.py                  # Other unit tests
```

---

## Core AI Modules

### 1. Universal Image Classifier (`ai/image_classifier/`)
- **Model**: EfficientNetV2-S (timm)
- **Classes**: 3 (Skin, Medicinal, Other)
- **Input**: 224×224 RGB
- **Features**: Ensemble inference, OOD detection (Energy/MSP/Entropy), Temperature scaling calibration
- **Checkpoint**: `ai/image_classifier/checkpoints/best_model.pth` (~244MB)

### 2. Skin Disease Classifier (`ai/training/`, `ai/preprocessing/`)
- **Model**: EfficientNetV2-S
- **Classes**: 22 skin diseases + "Unknown_Normal"
- **Input**: 256×256 RGB
- **Loss**: Focal Loss (γ=2.0) with class weights
- **Optimizer**: AdamW (lr=3e-4, wd=1e-4)
- **Scheduler**: ReduceLROnPlateau
- **Mixed Precision**: AMP (CUDA)
- **Pipeline**: Two-stage (Binary Healthy/Diseased → 22-class)
- **Explainability**: Grad-CAM on `conv_head` layer
- **OOD Detection**: Energy, MSP, Entropy, Combined
- **Checkpoint**: `ai/checkpoints/best_model.pth` (81MB)

### 3. Herb Identification (`ai/herb/`)
- **Model**: EfficientNetV2-S
- **Classes**: Determined by Medicinal_plant_dataset
- **Input**: 256×256 RGB
- **Features**: Binary leaf detector pre-filter, ArcFace metric learning, Ensemble, Calibration, OOD
- **Confidence Threshold**: 50%
- **Top-K**: 3 predictions
- **Checkpoint**: `ai/herb/checkpoints/best_model.pth` (243MB)

### 4. Knowledge Bases
- **Disease KB** (`ai/datasets/knowledge_base/disease_knowledge_base.json`): 22 diseases with description, symptoms, causes, prevention, self-care, when_to_consult_doctor, recommended_herbs
- **Herbal KB** (`ai/datasets/knowledge_base/herbal_knowledge_base.json`): Detailed herb info (botanical_name, family, active_compounds, phytochemicals, benefits, preparation_method, side_effects, contraindications, research_papers, skin_types, evidence_level)
- **Caching**: In-memory with hot-reload for development

### 5. LLM Integration (Ollama)
- **Model**: llama3.2:3b (local)
- **Client**: Async HTTP with retry (exponential backoff), circuit breaker, caching, fallback
- **Prompt Templates**: Skin disease summary, Herb summary, Chat context
- **Resilience**: Retry (3x), timeout (120s), fallback to cached responses

---

## Backend API Endpoints

### Authentication (`/api/v1/auth/`)
| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| POST | `/register` | ❌ | User registration |
| POST | `/login` | ❌ | Form-based login (OAuth2) |
| POST | `/login/json` | ❌ | JSON login |
| POST | `/refresh` | ✅ | Refresh access token |
| GET | `/me` | ✅ | Current user profile |
| PATCH | `/me` | ✅ | Update profile |
| POST | `/change-password` | ✅ | Change password |
| GET | `/users` | 👑 Admin | List users |
| GET | `/users/{id}` | 👑 Admin | Get user |
| PATCH | `/users/{id}` | 👑 Admin | Update user |
| DELETE | `/users/{id}` | 👑 Admin | Delete user |
| POST | `/validate` | ✅ | Token validation |

### Prediction (`/api/v1/predict/`)
| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| POST | `/` | ✅ | Universal prediction (skin/herb/other) |

### Herb Identification (`/api/v1/herb/`)
| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| POST | `/` | ✅ | Direct herb identification |

### LLM Services
| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| POST | `/summary/` | ✅ | AI summary generation |
| POST | `/chat/` | ✅ | Context-aware chatbot |

### Reports
| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| POST | `/report/` | ✅ | PDF report generation |

### Health & Monitoring
| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| GET | `/health` | ❌ | Liveness probe |
| GET | `/ready` | ❌ | Readiness probe (models, Ollama, disk) |
| GET | `/metrics` | ❌ | Prometheus metrics |

---

## Data Flow

```
┌─────────────┐
│  Frontend   │  User uploads image
└──────┬──────┘
       │ POST /api/v1/predict/ (multipart)
       ▼
┌─────────────┐
│  Backend    │  FastAPI receives upload
│  (FastAPI)  │  Validates file (magic bytes, MIME, size)
└──────┬──────┘
       │ classifier.predict(temp_path)
       ▼
┌─────────────────────────┐
│ Universal Classifier    │  3-class EfficientNetV2-S
│ (ai/image_classifier)   │  Returns: Skin/Medicinal/Other
└───────┬─────────────┘
            │
      ┌─────┴─────┐
      ▼           ▼
┌─────────┐ ┌─────────────┐
│  Skin   │ │  Medicinal  │
│ Pipeline│ │  Pipeline   │
└────┬────┘ └──────┬──────┘
     │             │
     ▼             ▼
Skin Disease    Herb ID Model
Model (22 cls)  (N classes)
     │             │
     ▼             ▼
Knowledge      Knowledge
Base Lookup    Base Lookup
     │             │
     ▼             ▼
AI Summary     AI Summary
(Ollama)       (Ollama)
     │             │
     └─────┬───────┘
           ▼
       JSON Response
           │
           ▼
    Frontend renders results
```

---

## Commands to Run

### Prerequisites
```bash
# 1. Install Ollama and pull model
ollama pull llama3.2:3b
ollama serve  # Run in background

# 2. Python environment (3.11+)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or venv\Scripts\activate  # Windows

# 3. Install dependencies
pip install -r requirements.txt
pip install -r backend/requirements.txt

# 4. Frontend dependencies
cd frontend
npm install
cd ..
```

### Development
```bash
# Terminal 1: Backend
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Terminal 2: Frontend
cd frontend
npm run dev

# Terminal 3: Ollama (if not running as service)
ollama serve

# Code Quality Tools
npm run lint          # Check for code quality issues
npm run lint:fix      # Auto-fix fixable issues
npm run format        # Format code with Prettier
```

### Production
```bash
# Backend
cd backend
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4

# Frontend
cd frontend
npm run build
npm start

# Or with Docker
docker build -t herbal-ai-backend ./backend
docker build -t herbal-ai-frontend ./frontend
docker-compose up -d
```

### Testing
```bash
# All tests
python -m pytest tests/ -v

# Specific test files
python -m pytest tests/test_auth.py -v
python -m pytest tests/test_api_endpoints.py -v

# With coverage
python -m pytest tests/ --cov=backend --cov=ai --cov-report=html
```

### Training Models
```bash
# Skin Disease
cd ai/training
python train.py

# Herb
cd ai/herb/training
python train.py

# Universal Classifier
cd ai/image_classifier
python trainer.py
```

---

## Configuration (`.env`)

```env
# Application
APP_NAME=Herbal-AI API
APP_VERSION=1.0.0
DEBUG=false
ENVIRONMENT=production

# Server
HOST=0.0.0.0
PORT=8000
WORKERS=4

# CORS
CORS_ORIGINS=https://yourdomain.com,https://www.yourdomain.com
CORS_ALLOW_CREDENTIALS=false
CORS_ALLOW_METHODS=GET,POST
CORS_ALLOW_HEADERS=*

# File Upload
MAX_UPLOAD_SIZE_MB=10
ALLOWED_MIME_TYPES=image/jpeg,image/png,image/jpg
ALLOWED_EXTENSIONS=.jpg,.jpeg,.png

# Model Settings
SKIN_DISEASE_MODEL_NAME=tf_efficientnetv2_s
SKIN_DISEASE_NUM_CLASSES=22
SKIN_DISEASE_IMAGE_SIZE=256
HERB_MODEL_NAME=tf_efficientnetv2_s
HERB_IMAGE_SIZE=256
HERB_CONFIDENCE_THRESHOLD=0.5
HERB_TOP_K=3
UNIVERSAL_MODEL_NAME=tf_efficientnetv2_s
UNIVERSAL_NUM_CLASSES=3
UNIVERSAL_IMAGE_SIZE=224

# Device
DEVICE=auto
NUM_WORKERS=2
PIN_MEMORY=true
USE_AMP=true

# Ollama
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llama3.2:3b
OLLAMA_TIMEOUT_SECONDS=120
OLLAMA_MAX_RETRIES=3
OLLAMA_RETRY_BACKOFF=1.0

# Rate Limiting
RATE_LIMIT_REQUESTS_PER_MINUTE=60
RATE_LIMIT_BURST=10

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json
LOG_REQUESTS=true

# Monitoring
ENABLE_METRICS=true
METRICS_PATH=/metrics

# Auth
SECRET_KEY=your-super-secret-key-change-in-production-min-32-chars
ACCESS_TOKEN_EXPIRE_MINUTES=30
REFRESH_TOKEN_EXPIRE_DAYS=7
COOKIE_SECURE=true
COOKIE_SAMESITE=lax
```

---

## ✅ Implemented Code Quality Improvements

To enhance maintainability, catch errors early, and ensure consistent code quality throughout the Herbal-AI codebase, the following developer tooling has been implemented:

### ESLint & Prettier Setup
- **ESLint** (`eslintrc.js`): Static analysis tool that identifies problematic patterns, enforces coding standards, and catches potential bugs
- **Prettier** (`prettierrc`): Opinionated code formatter that enforces consistent styling
- **Integration**: ESLint runs first to find/logic errors, then Prettier formats, then ESLint verifies again

### Key Benefits for Medical AI Application
1. **Error Prevention**: Catches type-related bugs, security vulnerabilities, and logical errors before runtime
2. **Consistency**: Eliminates formatting debates and ensures uniform code appearance
3. **Medical Compliance**: Facilitates auditing and review processes for healthcare software
4. **Developer Efficiency**: Saves time on manual formatting and style nitpicking in pull requests
5. **Safety Critical**: Particularly important for medical applications where code errors could have serious consequences

### Specific Improvements Made
- **Type Safety**: Enforced strict TypeScript rules (no `any`, explicit return types, etc.)
- **Security**: Banned dangerous patterns like `eval()`, `__proto__`, and implicit globals
- **Best Practices**: Required braces for control structures, strict equality (`===`), etc.
- **Next.js Compliance**: Enforced use of Next.js Image component for optimization
- **React Standards**: Enforced proper hooks usage and JSX formatting
- **Formatting Standards**: Consistent spacing, quotes, semicolons, trailing commas, etc.

### Usage Instructions
```bash
# Check for code quality issues
npm run lint

# Automatically fix fixable issues
npm run lint:fix

# Format all code with Prettier
npm run format
```

### Files Added
- `.eslintrc.js` - ESLint configuration with TypeScript/React/Next.js rules
- `.prettierrc` - Prettier formatting standards
- `.eslintignore` - Files/directories to exclude from ESLint
- `.prettierignore` - Files/directories to exclude from Prettier
- Updated `frontend/package.json` - Added lint/format scripts and dev dependencies

---

## Security Features

1. **File Upload Validation**
   - Magic bytes verification (JPEG/PNG signatures)
   - MIME type cross-check
   - File size limit (10MB)
   - Filename sanitization (path traversal prevention)
   - Secure temp file naming

2. **Authentication**
   - Argon2 password hashing
   - JWT access tokens (30 min) + refresh tokens (7 days)
   - Role-based access control (USER, ADMIN, RESEARCHER)
   - HttpOnly cookies for browser clients

3. **Rate Limiting**
   - 60 requests/minute per IP (configurable)
   - Burst limit: 10 requests/10 seconds
   - Disabled in test environment

4. **CORS**
   - Restricted origins
   - Credentials disabled
   - Limited methods (GET, POST)

5. **Error Handling**
   - Structured error responses with codes
   - No stack traces in production
   - Request ID correlation

---

## Monitoring & Observability

1. **Structured Logging** (structlog)
   - JSON format in production
   - Request ID propagation
   - Request/response logging with duration
   - Error context with traceback

2. **Prometheus Metrics** (`/metrics`)
   - HTTP requests total
   - Inference latency
   - Model loaded status
   - Error rates

3. **Health Checks**
   - `/health` - Liveness (process alive)
   - `/ready` - Readiness (models, Ollama, disk space)

---

## Medical Disclaimer

> **Important**: Herbal-AI provides educational guidance only. It is NOT a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified dermatologist or healthcare professional for medical concerns. Herbal supplements are NOT medical treatments and may have side effects or interactions.

---

## Future Enhancements

- [ ] Model Registry (MLflow)
- [ ] Batch Inference API
- [ ] ONNX/TensorRT Optimization
- [ ] PostgreSQL + Alembic Migrations
- [ ] Multi-language Support (i18n)
- [ ] PWA Support
- [ ] Automated Model Drift Detection
- [ ] A/B Testing Framework
- [ ] WebSocket for Real-time Chat

---

## License

This project is for educational and research purposes. Please ensure compliance with local regulations for medical AI applications.

---

---
*Last Updated: 2026-08-25*
*Version: 1.1.0*
# Herbal-AI
to run backend :uvicorn backend.app.main:app --reload
to run frontend : cd frontend
                  npm run dev


                  User uploads image
                        │
                        ▼
                Universal Image Classifier
                        │
                ┌──────┼──────────────┐
                │      │              │
                ▼      ▼              ▼
                Skin  Medicinal      Other
                │       │              │
                ▼       ▼              ▼
                Skin   Leaf Model   Return:
                Model               "Unsupported Image"
                │       │
                ▼       ▼
                Recommendation Engine
                │
                ▼
                PDF + Response
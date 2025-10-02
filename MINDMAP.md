# Mind Map — Odyssey / Oodi Project

- Project Vision
  - Personalized Persian AI assistant for developers
  - Continual learning from user interactions
  - Strong focus: high-quality code generation & mathematical rigor

- Core Components
  - CLI Chat Interface
    - Interactive loop
    - Commands (/exit, /push, /status)
  - Model Management
    - Loader (base vs fine-tuned)
    - Tokenizer handling
    - Save / Load / Push to HF
  - Data Pipeline
    - seed_from_forum.txt (persona + domain)
    - persian_buffer.txt (user <> assistant)
    - data validation & cleaning
  - Fine-tuning Engine
    - Prepare combined dataset
    - Tokenization and block-making
    - Trainer + custom callbacks (early stop, LR scheduler)
  - Prompt Manager
    - system_prompt.txt (sealed)
    - dynamic prompt assembly (persona + few-shot)
  - Generation Utils
    - safe generation wrappers (top-k/top-p, temperature)
    - log-prob extraction, filtering, n-best
  - Evaluation & Math Tools
    - Perplexity, cross-entropy, token-level log-probs
    - Code-focused metrics (syntax check, simple exec tests)
    - Complexity analysis helpers
  - Developer Tools
    - Code formatter integration (black)
    - Unit tests (pytest)
    - CI-ready scripts

- Files & Structure
  - src/
  - tests/
  - docs/
  - tools/ (data, scripts)
  - packaging: zip & upload

- Roadmap (v0 → v1 → v2)
  - v0: interactive CLI + seeded fine-tune loop
  - v1: robust eval metrics + safe generation + CI
  - v2: web UI, multi-user support, dataset versioning

- Non-functional Requirements
  - Reproducibility (random seeds, deterministic tokenization)
  - Privacy (local-only buffer by default)
  - Extensibility & modular code

# Odyssey (Oodi) — Personalized Persian AI Assistant with Continual Fine-Tuning
<img src="[assets/oodi_logo.png](https://www.neutrinoweb.com/images/oodi/oodi-llm.png)" alt="Oodi Logo" width="200"/>
<p align="center">
  <img src="https://img.shields.io/badge/Model-GPT--2_Fa-blue?style=for-the-badge&logo=huggingface" alt="Model Badge"/>
  <img src="https://img.shields.io/badge/Learning-Continual_FT-red?style=for-the-badge" alt="Learning Badge"/>
  <img src="https://img.shields.io/badge/Language-Persian_%2F_Farsi-green?style=for-the-badge" alt="Language Badge"/>
</p>

## Overview

**Odyssey (aka "Oodi")** is a lightweight, Persian-language conversational assistant built on top of a GPT-2 Persian base model.  
It is designed for **personalized** interactions and **continual learning**: the system stores short-term conversation history, and periodically fine-tunes the local model on the collected interactions plus a seed dataset (e.g., curated forum content and a persona prompt).

This repository contains the scripts and minimal infrastructure to:

- Run an interactive chat loop (terminal-based).
- Append user ↔ assistant exchanges to a local buffer.
- Combine a seed file (persona + domain text) with the buffer and fine-tune the GPT-2 Persian model when enough data is collected.
- Optionally push the fine-tuned model to the Hugging Face Hub.

> **Note:** The project is intended as a personal / educational toolkit for experimenting with continual fine-tuning and building a small Persian assistant. It is **not** production-ready, and you should treat any generated content and model outputs accordingly.

---

## Key Features

- **Base model:** `HooshvareLab/gpt2-fa` (GPT-2 for Persian).  
- **Continual Fine-tuning:** Automatically fine-tunes after a configurable number of buffered exchanges (default: 20).  
- **Persona & Seed Source:** The script can scrape and incorporate forum/homepage content (e.g., JumpLander) as a seed dataset and prepend a persona prompt.  
- **Interactive CLI chat:** Simple terminal interface for conversational testing.  
- **Hugging Face integration:** Optional `push` command uploads model + tokenizer to HF Hub (requires `HF_TOKEN`).

---

## Architecture & Technical Details

- **Model type:** Decoder-only Transformer (GPT-2 architecture).  
- **Tokenization:** GPT-2 style BPE tokenizer (`GPT2TokenizerFast`). The code sets `pad_token` if missing.  
- **Training objective:** Causal language modeling (predict next token).  
- **Framework:** PyTorch + Hugging Face Transformers (`Trainer`).  
- **Fine-tuning strategy:** Concatenate seed + buffer into a single text file, tokenize, split into fixed-size blocks (MAX_LENGTH) and train using `Trainer`.  
- **Device support:** Uses CUDA if available; supports fp16 when GPU present.

### Typical hyperparameters in the included script
- `BATCH_SIZE = 2`  
- `EPOCHS = 1`  
- `LR = 5e-5`  
- `MAX_LENGTH = 128` (block size for tokens)  
- Sampling on generation: `top_k=50`, `top_p=0.95`, `temperature=0.8`

> These values are intentionally conservative for local experimentation. Increase batch size, epochs, and sequence length if you have more GPU memory.

---

## How It Works (Workflow)

1. On first run, the script checks for a seed file. If missing, it scrapes the configured forum URL to build a `seed_from_forum.txt` which includes a persona header and scraped content.  
2. The interactive chat loop starts, loading the base or previously fine-tuned model.  
3. Each user prompt and model response are appended to `persian_buffer.txt`.  
4. When the buffer reaches the configured minimum number of exchanges (default 20), the script:
   - Combines `seed_from_forum.txt` + `persian_buffer.txt` into a temporary training file.
   - Fine-tunes the GPT-2 model on that combined file via the `Trainer`.
   - Saves the updated model to `persian_gpt2_personal/`.
   - Clears the buffer so new interactions are collected for the next round.

---

## Repository Structure

```
odyssey/
│── persian_gpt2_personal/       # saved fine-tuned model & tokenizer (output)
│── persian_buffer.txt           # buffered conversations (user <> assistant)
│── seed_from_forum.txt          # initial persona + scraped content
│── your_script_name.py          # main script (chat loop, training, utils)
│── requirements.txt             # Python dependencies
│── README_Oodi_Odyssey.md       # this README file
```

---

## Prerequisites

- Python 3.8 or newer  
- Recommended: a CUDA-capable GPU with enough VRAM for fine-tuning  
- Environment variables:
  - `HF_TOKEN` — optional, required for pushing to Hugging Face Hub

Python packages (minimum):
- `torch`
- `transformers`
- `beautifulsoup4`
- `requests`
- `huggingface_hub`

Install via `requirements.txt` (example):
```bash
python -m venv venv
source venv/bin/activate   # Linux / macOS
# venv\Scripts\activate  # Windows (PowerShell)
pip install -r requirements.txt
```

---

## Running the Assistant

Start the interactive chat loop:

```bash
python your_script_name.py
```

Basic commands inside the chat loop:
- `/exit` or `/quit` — exit the program.  
- `/push username/repo_name` — upload the saved model & tokenizer to Hugging Face Hub (requires `HF_TOKEN` set in your environment).

Example:
```bash
export HF_TOKEN="hf_xxx..."   # Linux / macOS
python your_script_name.py
# In the chat:
# /push your_username/oodi-model
```

---

## Configuration Tips

- `FORUM_URL`: Change the `FORUM_URL` constant in the script if you want to seed from a different website. Make sure scraping that site is allowed by its robots.txt and terms of service.  
- `MIN_BUFFER`: Controls the number of exchanges before automatic fine-tuning. Lower values mean more frequent small updates; higher values produce larger datasets per fine-tune.  
- `MAX_LENGTH`, `BATCH_SIZE`, `EPOCHS`: Tune these based on GPU resources and desired training behavior.

---

## Fine-tuning Implementation Notes

- The script tokenizes the combined seed+buffer text and forms examples by sliding a fixed-size block (`MAX_LENGTH`) and uses these as both inputs and labels for causal LM training.  
- DataCollatorForLanguageModeling is used with `mlm=False` (causal LM).  
- `Trainer` manages optimization, logging, and checkpointing. The script sets `save_total_limit` to limit checkpoints.

---

## Safety, Privacy & Ethical Considerations

- **Personal data:** The buffer stores user exchanges locally. Treat the buffer as sensitive — do not commit it to public repositories.  
- **Content moderation:** Outputs are not safety-filtered automatically. Consider adding content filters or moderation steps before exposing the model to untrusted users.  
- **Permissions:** Make sure you have permission to use and store any scraped content used as seed data.

---

## Contributing

Contributions, issues, and suggestions are welcome. For meaningful contributions:

1. Fork the repository.  
2. Open a feature branch.  
3. Create clear commits and a descriptive PR explaining the change and motivation.

---

## License

This project is provided under the **MIT License**. See `LICENSE` for full details.

---

## Contact & Attribution

- Repo link (example): `https://github.com/Osodyssey/odyssey`  
- Base Persian GPT-2 model used: `HooshvareLab/gpt2-fa`  
- Built for personal experimentation and research. If you share derived models or datasets, respect licenses and privacy concerns.

import os
import requests
from pathlib import Path
from bs4 import BeautifulSoup
import torch
from transformers import (
    GPT2LMHeadModel,
    GPT2TokenizerFast,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from huggingface_hub import login

MODEL_NAME = "HooshvareLab/gpt2-fa"
SAVE_DIR = Path("persian_gpt2_personal")
BUFFER_FILE = Path("persian_buffer.txt")
SEED_FILE = Path("seed_from_forum.txt") # نام فایل سید تغییر کرد تا مشخص باشد
FORUM_URL = "https://jumplander.org/?fa=forum" # URL هدف برای Scraping

MIN_BUFFER = 20
BATCH_SIZE = 2
EPOCHS = 1
LR = 5e-5
MAX_LENGTH = 128
device = "cuda" if torch.cuda.is_available() else "cpu"

def fetch_forum_index_text(url):
    """این تابع به طور خاص برای استخراج عناوین و توضیحات از صفحه اصلی انجمن JumpLander طراحی شده است."""
    try:
        resp = requests.get(url, timeout=15, headers={"User-Agent": "PersonalGPT2Bot/1.0"})
        resp.raise_for_status()
        print(f"[INFO] Successfully fetched {url}")
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Could not fetch {url}: {e}")
        return ""

    soup = BeautifulSoup(resp.text, "html.parser")
    texts = []

    for tag in soup(["script", "style", "noscript", "svg", "img"]):
        tag.decompose()

    forum_blocks = soup.find_all("div", class_="forabg")
    for block in forum_blocks:
        category_title_tag = block.find("span", class_="corners-top")
        if category_title_tag and category_title_tag.span:
             texts.append(f"## دسته: {category_title_tag.span.get_text(strip=True)}")

        forum_rows = block.find_all("li", class_="row")
        for row in forum_rows:
            title_tag = row.find("a", class_="forumtitle")
            desc_tag = row.find("div", class_="forum-description")
            if title_tag:
                title = title_tag.get_text(strip=True)
                description = desc_tag.get_text(strip=True) if desc_tag else "بدون توضیحات."
                texts.append(f"### انجمن: {title}\n- {description}")

    return "\n\n".join(texts)

def build_seed_from_forum():
    """فایل سید اولیه را با استفاده از محتوای اسکرپ شده از انجمن و متن شخصیت (Persona) می‌سازد."""
    print(f"[INFO] Scraping forum summary from {FORUM_URL}...")
    forum_content = fetch_forum_index_text(FORUM_URL)

    if not forum_content:
        print("[WARN] No content scraped. Seed file will only contain the persona.")

    persona = (
        "###PERSONA###\n"
        "You are a Persian programming assistant specialized for the user Osodyssey / JumpLander. "
        "You answer in Persian, provide clear code examples, and adopt a concise professional tone. "
        "When asked for code, return runnable snippets enclosed in ```python``` or appropriate language fences.\n\n"
    )

    full_text = persona + f"###SOURCE###\n{FORUM_URL}\n###CONTENT###\n{forum_content}\n\n"
    SEED_FILE.write_text(full_text, encoding="utf-8")
    print(f"[INFO] Seed file written: {SEED_FILE} ({len(full_text)} chars)")

def ensure_files():
    SAVE_DIR.mkdir(exist_ok=True)
    if not BUFFER_FILE.exists():
        BUFFER_FILE.write_text("", encoding="utf-8")
    if not SEED_FILE.exists():
        build_seed_from_forum()

def load_model_and_tokenizer(model_dir=None):
    if model_dir and Path(model_dir).exists() and any(Path(model_dir).iterdir()):
        print(f"[INFO] Loading fine-tuned model from local dir: {model_dir}")
        model = GPT2LMHeadModel.from_pretrained(model_dir)
        tokenizer = GPT2TokenizerFast.from_pretrained(model_dir)
    else:
        print(f"[INFO] Loading base model from Hugging Face: {MODEL_NAME}")
        model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
        tokenizer = GPT2TokenizerFast.from_pretrained(MODEL_NAME)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.to(device)
    return model, tokenizer

def generate_reply(model, tokenizer, prompt, max_new_tokens=150):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.8,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    return text[len(prompt):].strip()

def append_to_buffer(prompt, response):
    line = f"###PROMPT###\n{prompt}\n###RESPONSE###\n{response}\n\n"
    with open(BUFFER_FILE, "a", encoding="utf-8") as f:
        f.write(line)

def buffer_length() -> int:
    txt = BUFFER_FILE.read_text(encoding="utf-8")
    return txt.count("###PROMPT###")

def prepare_combined_training_file(tmp="tmp_combined.txt"):
    seed = SEED_FILE.read_text(encoding="utf-8") if SEED_FILE.exists() else ""
    buf = BUFFER_FILE.read_text(encoding="utf-8")
    combined = seed + "\n\n" + buf
    Path(tmp).write_text(combined, encoding="utf-8")
    return tmp

def fine_tune_on_file(model, tokenizer, file_path):
    text = Path(file_path).read_text(encoding="utf-8")
    
    if not text.strip():
        print("[WARN] Training file is empty. Skipping fine-tuning.")
        return

    from torch.utils.data import Dataset

    class TextBlockDataset(Dataset):
        def __init__(self, tokenizer, text, block_size):
            tokenized_text = tokenizer(text, truncation=False)["input_ids"]
            self.examples = []
            for i in range(0, len(tokenized_text) - block_size + 1, block_size):
                self.examples.append(tokenized_text[i : i + block_size])

        def __len__(self):
            return len(self.examples)

        def __getitem__(self, i):
            item = torch.tensor(self.examples[i], dtype=torch.long)
            return {"input_ids": item, "labels": item.clone()}

    dataset = TextBlockDataset(tokenizer, text, MAX_LENGTH)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    training_args = TrainingArguments(
        output_dir=str(SAVE_DIR),
        overwrite_output_dir=True,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        learning_rate=LR,
        logging_steps=10,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
    )
    
    trainer.train()
    trainer.save_model(str(SAVE_DIR))
    tokenizer.save_pretrained(str(SAVE_DIR))

def push_to_hf(model_dir, repo_name, hf_token_env="HF_TOKEN"):
    token = os.environ.get(hf_token_env)
    if not token:
        print(f"[WARN] HF token not found in environment variable '{hf_token_env}'")
        return
    login(token=token)
    
    model = GPT2LMHeadModel.from_pretrained(model_dir)
    tokenizer = GPT2TokenizerFast.from_pretrained(model_dir)
    print(f"[INFO] Pushing model & tokenizer to Hugging Face Hub: {repo_name}")
    model.push_to_hub(repo_name)
    tokenizer.push_to_hub(repo_name)
    print(f"[INFO] Pushed successfully. You can now use the model from '{repo_name}'")

def run_chat_loop():
    ensure_files()
    model, tokenizer = load_model_and_tokenizer(SAVE_DIR if SAVE_DIR.exists() and any(SAVE_DIR.iterdir()) else None)
    
    print("="*50)
    print(f"✅ ربات فارسی شخصی آماده (device={device})")
    print("   - برای خروج، تایپ کنید /exit")
    print("   - برای آپلود مدل در هاگینگ‌فیس، تایپ کنید /push username/repo_name")
    print("="*50)
    
    try:
        while True:
            prompt = input("\n👤 شما: ").strip()
            if not prompt:
                continue
            if prompt.lower() == "/exit" or prompt.lower() == "/quit":
                break
            if prompt.lower().startswith("/push"):
                parts = prompt.split()
                if len(parts) >= 2:
                    repo = parts[1]
                    push_to_hf(str(SAVE_DIR), repo)
                else:
                    print("[ERROR] Usage: /push your_hf_username/your_repo_name")
                continue

            response = generate_reply(model, tokenizer, prompt)
            print("🤖 ربات:", response)

            append_to_buffer(prompt, response)

            if buffer_length() >= MIN_BUFFER:
                print("\n[INFO] Buffer is full. Preparing to fine-tune the model...")
                tmp_file = prepare_combined_training_file()
                
                print("[INFO] Starting fine-tuning process (this may take a while)...")
                model, tokenizer = load_model_and_tokenizer(SAVE_DIR)
                fine_tune_on_file(model, tokenizer, tmp_file)
                
                print("[INFO] Fine-tuning complete. Model has been updated.")
                BUFFER_FILE.write_text("", encoding="utf-8")
                print("[INFO] Buffer has been cleared.")
    
    except KeyboardInterrupt:
        print("\n👋 خدانگهدار!")

if __name__ == "__main__":
    run_chat_loop()

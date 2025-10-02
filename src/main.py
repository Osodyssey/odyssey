"""Entry point for Oodi CLI chat and orchestration."""
from pathlib import Path
from src.prompt_manager import PromptManager
from src.model_loader import ModelLoader
from src.trainer import FineTuner
from src.generate import Generator
from src.data_utils import BufferManager, prepare_training_file

import argparse

ROOT = Path(__file__).resolve().parents[1]

def run_cli():
    pm = PromptManager(ROOT / 'system_prompt.txt')
    ml = ModelLoader(ROOT / 'persian_gpt2_personal')
    gm = Generator(ml.model, ml.tokenizer)
    bm = BufferManager(ROOT / 'persian_buffer.txt', ROOT / 'seed_from_forum.txt')
    ft = FineTuner(ml, bm)

    print("Oodi — Persian AI assistant (type /exit to quit)")
    while True:
        try:
            prompt = input("\n👤 You: ").strip()
            if not prompt:
                continue
            if prompt.lower().startswith('/exit'):
                break
            if prompt.lower().startswith('/push'):
                parts = prompt.split()
                if len(parts) >= 2:
                    repo = parts[1]
                    ft.push_to_hf(repo)
                else:
                    print("Usage: /push username/repo")
                continue
            system_prompt = pm.get_system_prompt()
            full_prompt = pm.assemble_prompt(system_prompt, prompt)
            response = gm.generate_reply(full_prompt)
            print("\n🤖 Oodi:", response)
            bm.append(prompt, response)
            if bm.length() >= bm.min_buffer:
                print("[INFO] Buffer reached threshold. Starting fine-tune cycle...")
                tmp = prepare_training_file(ROOT / 'seed_from_forum.txt', ROOT / 'persian_buffer.txt', ROOT / 'tmp_combined.txt')
                ft.fine_tune(tmp)
                bm.clear()
        except KeyboardInterrupt:
            print('\nGoodbye!')
            break

if __name__ == '__main__':
    run_cli()

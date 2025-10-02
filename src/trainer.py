"""Fine-tuning module with clean, testable functions."""
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from pathlib import Path
import torch
from typing import Optional
from src.model_loader import ModelLoader

class FineTuner:
    def __init__(self, model_loader: ModelLoader, buffer_manager=None, save_dir: Path = Path('persian_gpt2_personal')):
        self.ml = model_loader
        self.buffer_manager = buffer_manager
        self.save_dir = Path(save_dir)

    def _build_dataset(self, tokenizer, text: str, block_size: int):
        tokens = tokenizer(text, truncation=False)['input_ids']
        examples = []
        for i in range(0, max(1, len(tokens) - block_size + 1), block_size):
            examples.append(tokens[i:i+block_size])
        import torch
        class _Dataset(torch.utils.data.Dataset):
            def __init__(self, examples):
                self.examples = examples
            def __len__(self):
                return len(self.examples)
            def __getitem__(self, i):
                t = torch.tensor(self.examples[i], dtype=torch.long)
                return {'input_ids': t, 'labels': t.clone()}
        return _Dataset(examples)

    def fine_tune(self, combined_file: Path, epochs: int = 1, batch_size: int = 2, lr: float = 5e-5, block_size: int = 128):
        text = Path(combined_file).read_text(encoding='utf-8')
        if not text.strip():
            print('[WARN] Training file empty. Skipping.')
            return
        tokenizer = self.ml.tokenizer
        model = self.ml.model
        dataset = self._build_dataset(tokenizer, text, block_size)
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
        training_args = TrainingArguments(
            output_dir=str(self.save_dir),
            overwrite_output_dir=True,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            learning_rate=lr,
            logging_steps=10,
            save_total_limit=2,
            fp16=torch.cuda.is_available(),
        )
        trainer = Trainer(model=model, args=training_args, data_collator=data_collator, train_dataset=dataset)
        trainer.train()
        trainer.save_model(str(self.save_dir))
        tokenizer.save_pretrained(str(self.save_dir))
        print('[INFO] Fine-tune finished and saved to', self.save_dir)

    def push_to_hf(self, repo_name: str):
        from huggingface_hub import login
        import os
        token = os.environ.get('HF_TOKEN')
        if not token:
            print('[WARN] HF_TOKEN not set in environment.')
            return
        login(token=token)
        self.ml.model.push_to_hub(repo_name)
        self.ml.tokenizer.push_to_hub(repo_name)

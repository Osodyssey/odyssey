"""Model loader: encapsulates loading base or fine-tuned model and tokenizer."""
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from pathlib import Path
import torch

class ModelLoader:
    def __init__(self, model_dir: Path, base_name: str = 'HooshvareLab/gpt2-fa'):
        self.model_dir = Path(model_dir)
        self.base_name = base_name
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model, self.tokenizer = self._load()

    def _load(self):
        if self.model_dir.exists() and any(self.model_dir.iterdir()):
            model = GPT2LMHeadModel.from_pretrained(str(self.model_dir))
            tokenizer = GPT2TokenizerFast.from_pretrained(str(self.model_dir))
        else:
            model = GPT2LMHeadModel.from_pretrained(self.base_name)
            tokenizer = GPT2TokenizerFast.from_pretrained(self.base_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model.to(self.device)
        return model, tokenizer

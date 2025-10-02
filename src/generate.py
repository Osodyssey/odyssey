"""Generation utilities with probability scoring and safe wrappers."""
from transformers import PreTrainedModel, PreTrainedTokenizerFast
from typing import List
import torch
import math

class Generator:
    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizerFast, device: str = None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

    def generate_reply(self, prompt: str, max_new_tokens: int = 150, top_k: int = 50, top_p: float = 0.95, temperature: float = 0.8) -> str:
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)
        out = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )
        text = self.tokenizer.decode(out[0], skip_special_tokens=True)
        if prompt in text:
            return text[len(prompt):].strip()
        return text.strip()

    def get_token_logprobs(self, text: str) -> List[float]:
        inputs = self.tokenizer(text, return_tensors='pt').to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs, labels=inputs['input_ids'])
            logits = outputs.logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = inputs['input_ids'][..., 1:].contiguous()
            log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
            token_logps = []
            for i in range(shift_labels.size(1)):
                lbl = shift_labels[0, i].item()
                token_logps.append(log_probs[0, i, lbl].item())
        return token_logps

    def perplexity(self, text: str) -> float:
        inputs = self.tokenizer(text, return_tensors='pt').to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs, labels=inputs['input_ids'])
            loss = outputs.loss.item()
        return math.exp(loss)

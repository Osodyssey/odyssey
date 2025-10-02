from pathlib import Path
from typing import Tuple

class BufferManager:
    def __init__(self, buffer_path: Path, seed_path: Path, min_buffer: int = 20):
        self.buffer_path = Path(buffer_path)
        self.seed_path = Path(seed_path)
        self.min_buffer = min_buffer
        if not self.buffer_path.exists():
            self.buffer_path.write_text('', encoding='utf-8')
        if not self.seed_path.exists():
            self.seed_path.write_text('# Seed file (create manually or via scraper)\n', encoding='utf-8')

    def append(self, prompt: str, response: str):
        line = f"###PROMPT###\n{prompt}\n###RESPONSE###\n{response}\n\n"
        self.buffer_path.write_text(self.buffer_path.read_text(encoding='utf-8') + line, encoding='utf-8')

    def length(self) -> int:
        txt = self.buffer_path.read_text(encoding='utf-8')
        return txt.count('###PROMPT###')

    def clear(self):
        self.buffer_path.write_text('', encoding='utf-8')

def prepare_training_file(seed_path: Path, buffer_path: Path, out_path: Path = None) -> Path:
    seed = seed_path.read_text(encoding='utf-8') if seed_path.exists() else ''
    buf = buffer_path.read_text(encoding='utf-8') if buffer_path.exists() else ''
    combined = seed + '\n\n' + buf
    out = out_path or Path('tmp_combined.txt')
    out.write_text(combined, encoding='utf-8')
    return out

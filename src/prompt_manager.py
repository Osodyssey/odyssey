from pathlib import Path
from typing import Optional

class PromptManager:
    def __init__(self, system_prompt_path: Path):
        self.system_prompt_path = Path(system_prompt_path)
        self._system = self.system_prompt_path.read_text(encoding='utf-8') if self.system_prompt_path.exists() else ''

    def get_system_prompt(self) -> str:
        return self._system

    def assemble_prompt(self, system_prompt: str, user_input: str, few_shot: Optional[str]=None) -> str:
        parts = [system_prompt.strip()]
        if few_shot:
            parts.append('\n--- Few-shot examples:\n' + few_shot.strip())
        parts.append('\n###USER\n' + user_input.strip())
        parts.append('\n###ASSISTANT\n')
        return '\n'.join(parts)

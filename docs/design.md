# Design Notes

This document outlines the modular design for Odyssey / Oodi project.
- Modular src/ with separation of concerns
- PromptManager handles system prompt and assembly
- Generator wraps generation and analysis utilities
- FineTuner encapsulates training logic and push-to-hf
- BufferManager handles buffering and persistence
- Utilities include math helpers and code validators

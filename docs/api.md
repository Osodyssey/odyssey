# API & Developer Notes

Key modules and their public methods:

- PromptManager(system_prompt_path)
  - get_system_prompt()
  - assemble_prompt(system_prompt, user_input, few_shot=None)

- ModelLoader(model_dir)
  - model, tokenizer attributes

- Generator(model, tokenizer)
  - generate_reply(prompt, ...)
  - get_token_logprobs(text)
  - perplexity(text)

- FineTuner(model_loader, buffer_manager)
  - fine_tune(combined_file, epochs=1, ...)
  - push_to_hf(repo_name)

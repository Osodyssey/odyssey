def test_system_prompt_exists():
    from pathlib import Path
    assert Path('system_prompt.txt').exists()

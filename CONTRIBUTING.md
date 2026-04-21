# Contributing Guide

Thanks for your interest in contributing to this project.

## How to Contribute

- Fork the repository and create your branch from `main`.
- Keep changes focused and small.
- Add clear commit messages.
- Run local checks before opening a PR.

## Local Setup

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Set API key:

   ```powershell
   $env:DASHSCOPE_API_KEY="your_key"
   ```

3. Run app:

   ```bash
   streamlit run app.py
   ```

## Coding Standards

- Follow PEP 8 style for Python code.
- Use meaningful names for functions and variables.
- Keep functions short and single-purpose.
- Add comments only where logic is non-obvious.

## Pull Request Checklist

- [ ] Code runs locally without errors
- [ ] No hardcoded secrets or API keys
- [ ] Documentation updated (if behavior changed)
- [ ] Changes are limited to the scope of the PR

## Security

- Never commit `.env`, API keys, or local vector DB data.
- If a key is exposed, revoke and regenerate it immediately.

IMPORTANT: Ensure you've thoroughly reviewed the [AGENTS.md](AGENTS.md) file before beginning any work.

## Personal Notes

- Forked for learning purposes and local experimentation
- Keep API keys out of version control (use `.env` file)
- Run tests before committing any changes
- Prefer `claude-3-5-sonnet` model for most tasks (better cost/performance tradeoff)
- Use `--verbose` flag when debugging unexpected outputs
- Set default max_tokens to 2048 (upstream default of 1024 is too low for longer responses)

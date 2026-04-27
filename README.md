# Free Claude Code

A free, open-source implementation of Claude Code — an AI-powered coding assistant that runs in your terminal.

> Fork of [Alishahryar1/free-claude-code](https://github.com/Alishahryar1/free-claude-code)

## Features

- 🤖 AI-powered code generation and editing
- 💬 Interactive terminal-based chat interface
- 📁 File system awareness and manipulation
- 🔍 Code search and analysis
- 🚀 Multiple Claude model support
- 🔑 Bring your own API key (Anthropic or compatible providers)

## Requirements

- Python 3.11+
- An Anthropic API key (or compatible provider)

## Installation

### From source

```bash
git clone https://github.com/your-username/free-claude-code.git
cd free-claude-code
pip install -e .
```

### Using pip (coming soon)

```bash
pip install free-claude-code
```

## Configuration

1. Copy the example environment file:

```bash
cp .env.example .env
```

2. Edit `.env` and add your API key:

```env
ANTHROPIC_API_KEY=your_api_key_here
```

See `.env.example` for all available configuration options.

## Usage

```bash
# Start an interactive session
free-claude-code

# Run with a specific model
free-claude-code --model claude-3-5-sonnet-20241022

# Run in a specific directory
free-claude-code --cwd /path/to/your/project
```

## Development

### Setup

```bash
pip install -e ".[dev]"
```

### Running tests

```bash
pytest
```

### Linting

```bash
ruff check .
ruff format .
```

## Contributing

Contributions are welcome! Please read [AGENTS.md](AGENTS.md) for guidelines on contributing to this project.

1. Fork the repository
2. Create a feature branch (`git checkout -b feat/amazing-feature`)
3. Commit your changes (`git commit -m 'feat: add amazing feature'`)
4. Push to the branch (`git push origin feat/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- Original project by [Alishahryar1](https://github.com/Alishahryar1/free-claude-code)
- Powered by [Anthropic's Claude](https://www.anthropic.com/claude)

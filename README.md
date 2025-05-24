# Browser Agent

A Python-based browser automation agent that can interact with web pages using Selenium.

## Setup

1. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On macOS/Linux
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
Browser_agent/
├── src/
│   ├── __init__.py
│   ├── agent.py
│   └── utils.py
├── tests/
│   ├── __init__.py
│   └── test_agent.py
├── .env.example
├── requirements.txt
└── README.md
```

## Usage

1. Copy `.env.example` to `.env` and fill in your configuration
2. Import and use the agent in your code:

```python
from src.agent import BrowserAgent

agent = BrowserAgent()
agent.navigate("https://example.com")
```

## Testing

Run tests using pytest:
```bash
pytest tests/
```
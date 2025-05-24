# Web Perception Agent

A web automation agent that uses Together.ai's vision models to perceive and interact with web pages.

## Features

- Visual perception using Together.ai's vision models
- Automated web interactions (click, type, scroll, etc.)
- Screenshot capture and analysis
- Gradio web interface
- Retry mechanisms and error handling
- Progress tracking and status updates

## Requirements

- Python 3.8+
- Together.ai API key
- Playwright browsers
- Required Python packages (see requirements.txt)

## Setup

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-name>
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Install Playwright browsers:
   ```bash
   playwright install
   ```

4. Configure Together.ai API key:
   - Open `config.py`
   - Replace `YOUR-TOGETHER-API-KEY-HERE` with your actual Together.ai API key

## Usage

1. Start the Gradio interface:
   ```bash
   python main.py
   ```

2. Access the web interface (default: http://localhost:7860)

3. Enter a URL and task description:
   - Example: "Navigate to wikipedia.org, type 'climate change' in the search box, and scroll to the bottom of the page"
   - The agent will execute the task using visual perception and web automation

## Available Actions

The agent can perform the following actions:
1. Navigate to URLs
2. Click on elements
3. Type text
4. Scroll pages
5. Wait for elements
6. Login to websites
7. Select options
8. Hover over elements
9. Focus on elements
10. Submit forms

## Troubleshooting

If you encounter issues:

1. Check your Together.ai API key is correctly set in config.py
2. Ensure all dependencies are installed
3. Verify Playwright browsers are installed
4. Check your internet connection
5. Try running in non-headless mode for debugging
6. Check the debug logs in the interface

## License

MIT License 

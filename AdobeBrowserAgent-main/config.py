"""Configuration settings for the web agent."""

import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Together.ai Configuration
TOGETHER_API_KEY = "tgp_v1_2sOLORmNOYRP6CFcpVtamEyyqI27MAA0k27Ehnzka9c"  # Replace with your actual Together.ai API key

if not TOGETHER_API_KEY:
    raise ValueError(
        "Together.ai API key not configured. Please set your API key in config.py"
    )

# Model configuration
DEFAULT_CHAT_MODEL = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
DEFAULT_VISION_MODEL = "meta-llama/llama-vision-free"

# Agent configuration
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 1024
DEFAULT_STREAM = True

# Web agent configuration
DEFAULT_HEADLESS = False
DEFAULT_TIMEOUT = 30000  # milliseconds
MAX_RETRIES = 3
WAIT_TIME = 2  # seconds

# Retry settings
RETRY_DELAY = 1  # seconds 
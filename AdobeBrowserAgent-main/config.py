"""
Configuration settings for the Web Agent.
"""

import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Together.ai API settings
TOGETHER_API_KEY = "tgp_v1_90IItRNdPY_x27F1VC8c-PUUFvnR5CiwCv4ukbvDszk"  # Add your Together.ai API key here

if not TOGETHER_API_KEY:
    raise ValueError(
        "Together.ai API key not configured. Please set your API key in config.py"
    )

# Model configuration
DEFAULT_CHAT_MODEL = "togethercomputer/llama-2-70b-chat"
DEFAULT_VISION_MODEL = "togethercomputer/llama-2-70b-chat"

# Browser settings
DEFAULT_HEADLESS = False
DEFAULT_MAX_TOKENS = 1000

# Logging settings
DEBUG = True

# Agent configuration
DEFAULT_TEMPERATURE = 0.7
DEFAULT_STREAM = True

# Web agent configuration
DEFAULT_TIMEOUT = 30000  # milliseconds
MAX_RETRIES = 3
WAIT_TIME = 2  # seconds

# Retry settings
RETRY_DELAY = 1  # seconds 
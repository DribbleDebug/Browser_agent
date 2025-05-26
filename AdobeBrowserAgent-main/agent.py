"""
Web Agent with visual perception capabilities.

This module implements a web automation agent that can:
1. Navigate and interact with web pages
2. Take screenshots and analyze page content
3. Make intelligent decisions based on Together.ai's vision models
4. Execute common web actions like clicking, typing, and scrolling
5. Provide a Gradio-based user interface for interaction
"""

import base64
import json
import time
import os
import requests
from io import BytesIO
from typing import Dict, Any, Optional, Tuple, List, Generator
import re

from playwright.sync_api import sync_playwright
import gradio as gr
from PIL import Image

# Configuration settings
TOGETHER_API_KEY = "tgp_v1_90IItRNdPY_x27F1VC8c-PUUFvnR5CiwCv4ukbvDszk"  # Replace with your Together.ai API key
DEFAULT_CHAT_MODEL = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
DEFAULT_VISION_MODEL = "meta-llama/llama-vision-free"
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 1024
DEFAULT_STREAM = True
DEFAULT_HEADLESS = False
DEFAULT_TIMEOUT = 30000  # milliseconds
MAX_RETRIES = 3
WAIT_TIME = 2  # seconds
RETRY_DELAY = 1  # seconds

class TogetherClient:
    """Simplified Together.ai API client."""
    
    def __init__(self, api_key: str):
        """Initialize the Together.ai client."""
        self.api_key = api_key
        self.base_url = "https://api.together.xyz/v1"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    def chat_completions(
        self,
        messages: List[Dict[str, str]],
        model: str = DEFAULT_CHAT_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS
    ) -> Dict[str, Any]:
        """Send a chat completion request."""
        endpoint = f"{self.base_url}/chat/completions"
        
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False
        }
        
        try:
            response = requests.post(
                endpoint,
                headers=self.headers,
                json=payload
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(f"Together.ai API request failed: {str(e)}")

# Define the WebPerceptionAgent class
class WebPerceptionAgent:
    """An agent that perceives and interacts with web pages using Together.ai's vision models.
    
    This agent combines web automation (via Playwright) with AI perception (via Together.ai)
    to understand and interact with web pages intelligently. It can:
    - Navigate to URLs
    - Take and analyze screenshots
    - Execute common web actions (click, type, scroll)
    - Make decisions based on visual context
    - Maintain action history and state
    """
    
    def __init__(self, headless: bool = DEFAULT_HEADLESS):
        """Initialize the web perception agent."""
        # Browser automation settings
        self.headless = headless
        self.browser = None
        self.page = None
        self.context = None
        self.playwright = None
        
        # Initialize AI client
        try:
            self.together_client = TogetherClient(api_key=TOGETHER_API_KEY)
            print("✓ Together.ai client initialized")
        except Exception as e:
            print(f"Warning: Together client initialization failed: {str(e)}")
            self.together_client = None
            
        # State tracking
        self.last_action = None
        self.action_history = []
        self.screenshots = []
        self.current_status = ""
        
    def update_status(self, status: str):
        """Update the current status and notify any listeners.
        
        This method is used to keep track of the agent's current state
        and provide feedback to users/interfaces.
        """
        self.current_status = status
        print(f"Status: {status}")
        
    def take_screenshot(self, reason: str = ""):
        """Take a screenshot of the current page with context.
        
        Args:
            reason: Why the screenshot was taken
            
        Returns:
            dict: Screenshot data including:
                - image: PIL Image object
                - timestamp: When taken
                - reason: Why taken
                - url: Page URL
                - action: Last action performed
        """
        try:
            screenshot_buffer = self.page.screenshot()
            screenshot = {
                'image': Image.open(BytesIO(screenshot_buffer)),
                'timestamp': time.time(),
                'reason': reason,
                'url': self.page.url,
                'action': self.last_action
            }
            self.screenshots.append(screenshot)
            return screenshot
        except Exception as e:
            print(f"Warning: Screenshot failed: {str(e)}")
            return None

    def analyze_page_and_decide(self, task: str) -> Tuple[Dict[str, Any], str]:
        """Analyze the current page and decide on the next action."""
        try:
            # Get current page state
            state = self.get_page_state()
            task_lower = task.lower()  # Use lowercase only for comparison
            
            # Handle search tasks
            if any(word in task_lower for word in ["search", "find", "look up"]):
                # Extract search terms from original task
                search_terms = None
                for word in ["search", "find", "look up"]:
                    if word in task_lower:
                        search_terms = task.split(word, 1)[1].strip()  # Use original case
                        break
                
                if search_terms:
                    return {
                        "action": "search",
                        "text": search_terms  # Use original case
                    }, "Performing search"

            # Handle type tasks
            if "type" in task_lower:
                # Extract text to type from original task
                type_text = task.split("type", 1)[1].strip()  # Use original case
                
                # Check if this is a username/email field
                if "username" in type_text.lower() or "email" in type_text.lower() or "user" in type_text.lower():
                    return {
                        "action": "type",
                        "selector": """
                            input[type='text'],
                            input[type='email'],
                            input[name='username'],
                            input[name='email'],
                            input[id='username'],
                            input[id='email'],
                            input[name*='user'],
                            input[id*='user'],
                            input[aria-label*='username' i],
                            input[aria-label*='email' i],
                            input[placeholder*='username' i],
                            input[placeholder*='email' i],
                            .username-input,
                            .email-input,
                            #username,
                            #email,
                            input:not([type='submit']):not([type='button']):not([type='hidden']):not([type='password'])
                        """,
                        "text": type_text  # Use original case
                    }, "Typing into username/email field"
                # Check if this is a password field
                elif "password" in type_text.lower() or any(word in task_lower for word in ["pass", "pwd", "secret"]):
                    return {
                        "action": "type",
                        "selector": """
                            input[type='password'],
                            input[name='password'],
                            input[id='password'],
                            input[name*='pass'],
                            input[id*='pass'],
                            input[aria-label*='password' i],
                            input[placeholder*='password' i],
                            .password-input,
                            #password,
                            input[type='password']:not([disabled])
                        """,
                        "text": type_text,  # Use original case
                        "expect_navigation": False
                    }, "Typing into password field"
                else:
                    # Generic text input
                    return {
                        "action": "type",
                        "selector": """
                            input[type='text']",
                            input[type='search']",
                            input[type='email']",
                            input[name='q']",
                            input[name='search']",
                            input[placeholder*='search' i]",
                            input[placeholder*='type' i]",
                            input[placeholder*='enter' i]",
                            ".search-input",
                            ".text-input",
                            "textarea",
                            "input:not([type='submit']):not([type='button']):not([type='hidden'])"
                        """,
                        "text": type_text
                    }, "Typing text"
            
            # Handle blank page navigation
            if state['url'] == "about:blank":
                # Extract URL from task if present
                url_match = re.search(r'https?://[^\s<>"]+|www\.[^\s<>"]+', task)
                if url_match:
                    url = url_match.group(0)
                    if not url.startswith(('http://', 'https://')):
                        url = 'https://' + url
                    return {"action": "navigate", "text": url}, "Navigating to URL from task"
                else:
                    return {"action": "navigate", "text": "https://www.google.com"}, "Navigating to default URL"

            # Handle click tasks with position or text targeting
            if "click" in task_lower:
                # Extract text after "click"
                click_text = task_lower.split("click", 1)[1].strip()
                
                # Handle login/signin/continue/next button specifically
                if "login" in click_text.lower() or "sign in" in click_text.lower() or "continue" in click_text.lower() or "next" in click_text.lower():
                    return {
                        "action": "click",
                        "selector": """
                            button[type='submit'],
                            input[type='submit'],
                            button:text-is("Login"),
                            button:text-matches("Login", "i"),
                            button:text-is("Sign in"),
                            button:text-matches("Sign in", "i"),
                            a:text-is("Login"),
                            a:text-matches("Login", "i"),
                            a:text-is("Sign in"),
                            a:text-matches("Sign in", "i"),
                            [role='button']:text-is("Login"),
                            [role='button']:text-matches("Login", "i"),
                            [role='button']:text-is("Sign in"),
                            [role='button']:text-matches("Sign in", "i"),
                            input[value='Login'],
                            input[value='Sign in'],
                            .login-button,
                            .signin-button,
                            #login,
                            #signin,
                            button.radius,
                            button.fa-sign-in,
                            [class*='login'],
                            [class*='signin'],
                            [id*='login'],
                            [id*='signin'],
                            form button,
                            form input[type='submit'],
                            button:has-text("Login"),
                            button:has-text("Sign in"),
                            button:has-text("Log in"),
                            button:has-text("Submit"),
                            button:has-text("Continue"),
                            button:has-text("Next"),
                            [type='submit']
                        """,
                        "expect_navigation": True  # Signal that we expect navigation
                    }, "Clicking login/signin button with navigation"
                # Handle numerical positions (e.g., "click 3rd link")
                position_match = re.search(r'(?:link|button)?\s*(?:number|numbered|#)?\s*(\d+)(?:st|nd|rd|th)?(?:\s+(?:link|button))?|(\d+)(?:st|nd|rd|th)?\s+(?:link|button)', click_text)
                if position_match:
                    position = int(position_match.group(1) if position_match.group(1) else position_match.group(2))
                    return {
                        "action": "click",
                        "selector": f"""
                            a[href]:not([href='']):not([href^='#']):nth-child({position}),
                            a[href]:not([href='']):not([href^='#']):nth-of-type({position}),
                            a:nth-of-type({position}),
                            button:nth-of-type({position}),
                            [role='button']:nth-of-type({position}),
                            [role='link']:nth-of-type({position}),
                            .btn:nth-of-type({position}),
                            [data-testid*='button']:nth-of-type({position}),
                            [data-testid*='link']:nth-of-type({position}),
                            [class*='button']:nth-of-type({position}),
                            [class*='link']:nth-of-type({position})
                        """
                    }, f"Clicking element number {position}"
                # Handle special positions (first/last)
                elif "first" in click_text:
                    return {
                        "action": "click",
                        "selector": "a[href]:not([href='']):not([href^='#']):first-child, a[href]:not([href='']):not([href^='#'])"
                    }, "Clicking first valid link"
                elif "last" in click_text:
                    return {
                        "action": "click",
                        "selector": "a[href]:not([href='']):not([href^='#']):last-child, a[href]:not([href='']):not([href^='#']):last-of-type"
                    }, "Clicking last valid link"
                # Handle common button types
                elif "login" in click_text or "sign in" in click_text:
                    return {
                        "action": "click",
                        "selector": """
                            button[type='submit'],
                            input[type='submit'],
                            button:text-is("Login"),
                            button:text-matches("Login", "i"),
                            button:text-is("Sign in"),
                            button:text-matches("Sign in", "i"),
                            a:text-is("Login"),
                            a:text-matches("Login", "i"),
                            a:text-is("Sign in"),
                            a:text-matches("Sign in", "i"),
                            [role='button']:text-is("Login"),
                            [role='button']:text-matches("Login", "i"),
                            [role='button']:text-is("Sign in"),
                            [role='button']:text-matches("Sign in", "i"),
                            .login-button,
                            .signin-button,
                            #login,
                            #signin,
                            button.radius,
                            button.fa-sign-in,
                            [class*='login'],
                            [class*='signin'],
                            [id*='login'],
                            [id*='signin']
                        """
                    }, "Clicking login button"
                elif "submit" in click_text:
                    return {
                        "action": "click",
                        "selector": "button[type='submit'], input[type='submit'], button:contains('Submit'), .submit-button, #submit"
                    }, "Clicking submit button"
                elif "continue" in click_text:
                    return {
                        "action": "click",
                        "selector": """
                            button:text-is("Continue"),
                            button:text-matches("Continue", "i"),
                            [role='button']:text-is("Continue"),
                            [role='button']:text-matches("Continue", "i"),
                            input[type='submit'][value*='continue' i],
                            .btn:text-matches("Continue", "i"),
                            [data-testid*='continue' i],
                            [aria-label*='continue' i],
                            [title*='continue' i],
                            button:text-is("Next"),
                            button:text-matches("Next", "i"),
                            [role='button']:text-is("Next"),
                            [role='button']:text-matches("Next", "i"),
                            .btn:text-matches("Next", "i"),
                            [data-testid*='next' i],
                            [aria-label*='next' i],
                            [title*='next' i],
                            button[type='submit'],
                            input[type='submit']
                        """
                    }, "Clicking continue/next button"
                else:
                    # Try text-based clicking first (e.g., "click English" or "click on English")
                    text_match = re.search(r'(?:on\s+)?["\']?([^"\']+)["\']?$', click_text)
                    if text_match:
                        click_target = text_match.group(1).strip()
                        # Special handling for language links
                        if click_target.lower() in ["italiano", "english", "español", "deutsch", "français", "português", "日本語", "русский", "中文", "한국어"]:
                            return {
                                "action": "click",
                                "selector": f"""
                                    a[lang="{click_target}"],
                                    a[hreflang="{click_target}"],
                                    #p-lang a:text-is("{click_target}"),
                                    .interlanguage-link a:text-is("{click_target}"),
                                    .language-link:text-is("{click_target}"),
                                    [data-lang="{click_target}"],
                                    a:text-is("{click_target}"),
                                    a:text-matches("^{click_target}$", "i")
                                """
                            }, f"Clicking language link: {click_target}"
                        return {
                            "action": "click",
                            "selector": f"""
                                a:text-is("{click_target}"),
                                a:text-matches("{click_target}", "i"),
                                button:text-is("{click_target}"),
                                button:text-matches("{click_target}", "i"),
                                [role='button']:text-is("{click_target}"),
                                [role='button']:text-matches("{click_target}", "i"),
                                [role='link']:text-is("{click_target}"),
                                [role='link']:text-matches("{click_target}", "i"),
                                .btn:text-is("{click_target}"),
                                .btn:text-matches("{click_target}", "i"),
                                [data-testid*='button']:text-is("{click_target}"),
                                [data-testid*='button']:text-matches("{click_target}", "i"),
                                [data-testid*='link']:text-is("{click_target}"),
                                [data-testid*='link']:text-matches("{click_target}", "i"),
                                [aria-label*="{click_target}"],
                                [title*="{click_target}"],
                                a:contains('{click_target}'),
                                button:contains('{click_target}'),
                                [role='button']:contains('{click_target}'),
                                input[type='submit'][value*='{click_target}'],
                                [data-testid*='{click_target}'],
                                [aria-label*='{click_target}'],
                                .{click_target}-button,
                                #{click_target}
                            """
                        }, f"Clicking element with text: {click_target}"
                    
                    # Fallback to generic click by text content
                    click_text = click_text.strip("'\" ")
                    return {
                        "action": "click",
                        "selector": f"""
                            a:contains('{click_text}'),
                            button:contains('{click_text}'),
                            [role='button']:contains('{click_text}'),
                            input[type='submit'][value*='{click_text}'],
                            [data-testid*='{click_text}'],
                            [aria-label*='{click_text}'],
                            .{click_text}-button,
                            #{click_text}
                        """
                    }, f"Clicking element containing '{click_text}'"

            # Handle scroll tasks
            if "scroll" in task_lower:
                if "bottom" in task_lower:
                    return {"action": "scroll", "text": "bottom"}, "Scrolling to bottom of page"
                elif "top" in task_lower:
                    return {"action": "scroll", "text": "top"}, "Scrolling to top of page"
                else:
                    return {"action": "scroll", "text": "500"}, "Scrolling down"

            # If no specific action matched, use AI model for decision
            messages = [
                {
                    "role": "system",
                    "content": """You are a web automation assistant. Analyze the page and suggest the next action.
                    Available actions:
                    1. Type: {"action": "type", "selector": "CSS_SELECTOR", "text": "TEXT"}
                    2. Click: {"action": "click", "selector": "CSS_SELECTOR"}
                    3. Wait: {"action": "wait", "seconds": NUMBER}
                    4. Scroll: {"action": "scroll", "text": "DIRECTION"}
                    
                    IMPORTANT: Respond with ONLY a JSON object containing the next action."""
                },
                {
                    "role": "user",
                    "content": f"""Task: {task}
                    Current URL: {state['url']}
                    Page Title: {state['title']}
                    
                    What should be the next action? Respond with ONLY a JSON action object."""
                }
            ]
            
            # Get AI model's suggestion
            try:
                if not self.together_client:
                    self.together_client = TogetherClient(api_key=TOGETHER_API_KEY)
                
                response = self.together_client.chat_completions(
                    messages=messages,
                    model=DEFAULT_CHAT_MODEL,
                    temperature=DEFAULT_TEMPERATURE,
                    max_tokens=DEFAULT_MAX_TOKENS
                )
                
                if response and 'choices' in response and len(response['choices']) > 0:
                    content = response['choices'][0]['message']['content'].strip()
                    json_match = re.search(r'\{[^{]*\}', content)
                    if json_match:
                        try:
                            action_json = json.loads(json_match.group(0))
                            if self._validate_action(action_json):
                                return action_json, "Based on page analysis"
                        except json.JSONDecodeError:
                            pass
                
                return self._get_fallback_action(task)
                    
            except Exception as api_error:
                print(f"Together API error: {str(api_error)}")
                return self._get_fallback_action(task)
            
        except Exception as e:
            print(f"Error in analyze_page_and_decide: {str(e)}")
            return self._get_fallback_action(task)
    
    def _validate_action(self, action_data: Dict[str, Any]) -> bool:
        """Validate if the action data has all required fields."""
        if not isinstance(action_data, dict) or "action" not in action_data:
            return False
            
        action = action_data["action"].lower()
        
        # Validate action type and required fields
        if action == "navigate" and "text" in action_data:
            return True
        elif action == "click" and "selector" in action_data:
            return True
        elif action == "type" and "text" in action_data:
            # Make selector optional for type action
            return True
        elif action == "wait":
            # Either selector or seconds should be present
            return "selector" in action_data or "text" in action_data
            
        return False
    
    def setup(self):
        """Set up the browser and page."""
        try:
            # Initialize Playwright
            self.playwright = sync_playwright().start()
            self.browser = self.playwright.chromium.launch(headless=self.headless)
            self.context = self.browser.new_context()
            self.page = self.context.new_page()
            print("✓ Browser setup complete")
            
            # Test Together API
            try:
                if not self.together_client:
                    self.together_client = TogetherClient(api_key=TOGETHER_API_KEY)
                
                # Test with a simple chat completion
                response = self.together_client.chat_completions(
                    messages=[{
                        "role": "user",
                        "content": "Hello"
                    }],
                    model=DEFAULT_CHAT_MODEL,
                    max_tokens=10,
                    temperature=0.7
                )
                
                if response and 'choices' in response and len(response['choices']) > 0:
                    print("✓ Together API connection successful")
                else:
                    raise Exception("Invalid response format from Together API")
                    
            except Exception as api_error:
                raise Exception(f"Together API test failed: {str(api_error)}")
            
            return True
            
        except Exception as e:
            print(f"Setup failed: {str(e)}")
            self.cleanup()
            return False
    
    def _get_screenshot_url(self, screenshot_b64: str) -> str:
        """
        Save the screenshot locally and return a file URL.
        """
        try:
            # Create screenshots directory if it doesn't exist
            if not os.path.exists("screenshots"):
                os.makedirs("screenshots")
            
            # Save screenshot to file
            screenshot_path = os.path.join("screenshots", f"screenshot_{int(time.time())}.png")
            with open(screenshot_path, "wb") as f:
                f.write(base64.b64decode(screenshot_b64))
            
            # Return file URL
            return f"file://{os.path.abspath(screenshot_path)}"
        except Exception as e:
            print(f"Error saving screenshot: {str(e)}")
            return ""
        
    def get_page_state(self) -> Dict[str, Any]:
        """
        Capture the current page state (HTML and screenshot).
        
        Returns:
            A dictionary with HTML content and base64-encoded screenshot
        """
        # Get HTML content
        html_content = self.page.content()
        
        # Capture screenshot to memory buffer
        screenshot_buffer = self.page.screenshot()
        screenshot_b64 = base64.b64encode(screenshot_buffer).decode('utf-8')
        
        return {
            "html": html_content,
            "screenshot_b64": screenshot_b64,
            "url": self.page.url,
            "title": self.page.title
        }
    
    def execute_action(self, action_json_str: str) -> str:
        """Execute an action on the webpage with enhanced error handling and retries."""
        try:
            action_data = json.loads(action_json_str) if isinstance(action_json_str, str) else action_json_str
            
            # Update status
            self.update_status(f"Executing action: {action_data['action']}")
            
            # Take pre-action screenshot
            self.take_screenshot(f"Before {action_data['action']}")
            
            max_retries = 3
            retry_count = 0
            last_error = None
            
            while retry_count < max_retries:
                try:
                    # Wait for page to be ready
                    self.page.wait_for_load_state("domcontentloaded", timeout=5000)
                    
                    # Execute the action
                    result = self._execute_single_action(action_data)
                    
                    # Take post-action screenshot
                    self.take_screenshot(f"After {action_data['action']}")
                    
                    # Wait for any potential page updates
                    try:
                        self.page.wait_for_load_state("networkidle", timeout=5000)
                    except:
                        pass  # Ignore timeout on networkidle
                    
                    return result
                    
                except Exception as e:
                    last_error = e
                    retry_count += 1
                    print(f"Action attempt {retry_count} failed: {str(e)}")
                    
                    if retry_count < max_retries:
                        print(f"Retrying in 1 second...")
                        time.sleep(1)
                        
                        # Try to recover the page state
                        try:
                            self.page.wait_for_load_state("domcontentloaded", timeout=5000)
                        except:
                            pass
                    else:
                        print(f"Action failed after {max_retries} attempts")
                        break
            
            # If we get here, all retries failed
            error_msg = f"Action failed after {max_retries} retries: {str(last_error)}"
            print(error_msg)
            return error_msg
            
        except json.JSONDecodeError:
            error_msg = "Invalid action format: Not valid JSON"
            print(error_msg)
            return error_msg
        except Exception as e:
            error_msg = f"Error executing action: {str(e)}"
            print(error_msg)
            return error_msg
    
    def _execute_single_action(self, action_data: Dict[str, Any]) -> str:
        """Execute a single action with proper error handling."""
        action_type = action_data.get("action", "").lower()
        
        if action_type == "type" or action_type == "input" or action_type == "search":
            text = action_data.get("text", "")  # Get original text without converting to lower
            selector = action_data.get("selector", "")
            try:
                # Wait for page to be ready
                self.page.wait_for_load_state("domcontentloaded")
                time.sleep(0.5)  # Short wait for dynamic content

                # Handle direct type action with selector
                if selector:
                    try:
                        # Wait for element with increased timeout and retry
                        max_retries = 3
                        element = None
                        for attempt in range(max_retries):
                            try:
                                element = self.page.wait_for_selector(selector, state="visible", timeout=10000)
                                if element and element.is_visible():
                                    break
                            except:
                                if attempt < max_retries - 1:
                                    time.sleep(1)
                                    continue
                                raise
                        
                        if element:
                            # Ensure element is in view
                            element.scroll_into_view_if_needed()
                            time.sleep(0.2)
                            
                            # Click to focus
                            element.click()
                            time.sleep(0.2)
                            
                            # Clear existing text
                            element.fill("")
                            time.sleep(0.2)
                            
                            # Always use fill() to preserve exact case for all inputs
                            element.fill(text)  # Use original text without case conversion
                            time.sleep(0.2)
                            
                            return f"Successfully typed text into input field"
                    except Exception as type_error:
                        print(f"Type action failed: {str(type_error)}")
                        # Fall through to try other methods
                
                # Generic input handling
                input_selectors = [
                    "input[type='text']",
                    "input[type='search']",
                    "input[type='email']",
                    "input[name='q']",
                    "input[name='search']",
                    "input[placeholder*='search' i]",
                    "input[placeholder*='type' i]",
                    "input[placeholder*='enter' i]",
                    ".search-input",
                    ".text-input",
                    "textarea",
                    "input:not([type='submit']):not([type='button']):not([type='hidden'])"
                ]
                
                # Try each input selector
                for selector in input_selectors:
                    try:
                        input_element = self.page.locator(selector).first
                        if input_element and input_element.is_visible():
                            # Ensure element is in view
                            input_element.scroll_into_view_if_needed()
                            time.sleep(0.2)
                            
                            # Click to focus
                            input_element.click()
                            time.sleep(0.2)
                            
                            # Clear existing text
                            input_element.fill("")
                            time.sleep(0.2)
                            
                            # Always use fill() to preserve exact case for all inputs
                            input_element.fill(text)  # Use original text without case conversion
                            time.sleep(0.2)
                            
                            # For search actions, try to submit
                            if action_type == "search":
                                # Try to find and click a submit button
                                button_selectors = [
                                    "button[type='submit']",
                                    "input[type='submit']",
                                    "button.search-button",
                                    ".searchButton",
                                    "[aria-label*='search' i]",
                                    "button:has(svg)",  # Many search buttons have SVG icons
                                    "[type='submit']",
                                    "button:text-is('Search')",
                                    "button:text-matches('Search', 'i')"
                                ]
                                
                                button_clicked = False
                                for button_selector in button_selectors:
                                    try:
                                        button = self.page.locator(button_selector).first
                                        if button and button.is_visible():
                                            button.click()
                                            button_clicked = True
                                            break
                                    except:
                                        continue
                                
                                if not button_clicked:
                                    input_element.press("Enter")
                                
                                time.sleep(1)
                                try:
                                    self.page.wait_for_load_state("networkidle", timeout=5000)
                                except:
                                    pass
                                
                                return "Successfully submitted search"
                            else:
                                return f"Successfully typed text into input field"
                    except Exception as e:
                        print(f"Input selector {selector} failed: {str(e)}")
                        continue
                
                return "Could not find input field"
                
            except Exception as e:
                return f"Error performing action: {str(e)}"
                
        elif action_type == "click":
            try:
                # Define comprehensive click target selectors
                click_selectors = {
                    'button': [
                        "button",
                        "[role='button']",
                        ".btn",
                        "input[type='submit']",
                        "input[type='button']",
                        "[data-testid*='button']",
                        "[class*='button']",
                        "[id*='button']"
                    ],
                    'link': [
                        "a",
                        "[role='link']",
                        "[onclick]",
                        "[href]",
                        "[data-testid*='link']"
                    ],
                    'menu': [
                        "[role='menuitem']",
                        ".menu-item",
                        ".nav-item",
                        "[data-testid*='menu']",
                        "[class*='menu']"
                    ],
                    'tab': [
                        "[role='tab']",
                        ".tab",
                        "[data-testid*='tab']"
                    ],
                    'checkbox': [
                        "input[type='checkbox']",
                        "[role='checkbox']"
                    ],
                    'radio': [
                        "input[type='radio']",
                        "[role='radio']"
                    ]
                }
                
                target_text = action_data.get("text", "")  # Keep original case
                selector = action_data.get("selector", "")
                expect_navigation = action_data.get("expect_navigation", False)
                
                def handle_navigation(element, href=None, selector_type=None):
                    """Handle navigation after clicking an element."""
                    try:
                        # First try clicking with navigation wait
                        with self.page.expect_navigation(timeout=30000) as nav:
                            # Try multiple click methods in sequence
                            try:
                                # Try regular click first
                                element.click()
                            except:
                                try:
                                    # Try JavaScript click
                                    self.page.evaluate("(element) => element.click()", element)
                                except:
                                    try:
                                        # Try clicking center of element
                                        bbox = element.bounding_box()
                                        if bbox:
                                            self.page.mouse.click(
                                                bbox["x"] + bbox["width"] / 2,
                                                bbox["y"] + bbox["height"] / 2
                                            )
                                    except:
                                        # If all click methods fail, try direct navigation
                                        if href:
                                            self.page.goto(href, wait_until="domcontentloaded")
                                            return True, f"Navigation completed via direct URL: {href}"
                                        else:
                                            raise Exception("All click methods failed")
                            
                            # Wait for navigation to complete
                            nav.value
                        
                        # Additional wait for page stability
                        try:
                            self.page.wait_for_load_state("domcontentloaded", timeout=5000)
                            self.page.wait_for_load_state("networkidle", timeout=5000)
                        except:
                            pass
                        
                        desc = f"Successfully clicked{' ' + selector_type if selector_type else ''} and navigated to: {self.page.url}"
                        return True, desc
                        
                    except Exception as nav_error:
                        print(f"Navigation error: {str(nav_error)}")
                        
                        # If navigation fails, try fallback approaches
                        try:
                            # Try direct navigation if we have href
                            if href:
                                print(f"Trying direct navigation to: {href}")
                                self.page.goto(href, wait_until="domcontentloaded")
                                try:
                                    self.page.wait_for_load_state("networkidle", timeout=5000)
                                except:
                                    pass
                                return True, f"Navigation completed via fallback: {href}"
                            
                            # Otherwise try clicking without navigation wait
                            print("Trying click without navigation wait...")
                            try:
                                # Try regular click
                                element.click()
                            except:
                                try:
                                    # Try JavaScript click
                                    self.page.evaluate("(element) => element.click()", element)
                                except:
                                    try:
                                        # Try clicking center of element
                                        bbox = element.bounding_box()
                                        if bbox:
                                            self.page.mouse.click(
                                                bbox["x"] + bbox["width"] / 2,
                                                bbox["y"] + bbox["height"] / 2
                                            )
                                    except Exception as e:
                                        print(f"All click methods failed: {str(e)}")
                                        return False, "Could not click element"
                            
                            # Wait longer for potential dynamic navigation
                            time.sleep(3)
                            
                            try:
                                self.page.wait_for_load_state("domcontentloaded", timeout=5000)
                                return True, f"Navigation completed via click: {self.page.url}"
                            except:
                                return False, "Clicked element, waiting for navigation"
                                
                        except Exception as goto_error:
                            print(f"All navigation attempts failed: {str(goto_error)}")
                            return False, f"Navigation failed: {str(goto_error)}"
                
                # If a specific selector is provided, try that first
                if selector:
                    try:
                        # Wait for element with increased timeout
                        element = self.page.wait_for_selector(selector, state="visible", timeout=30000)
                        if element:
                            # Scroll element into view and wait a moment
                            element.scroll_into_view_if_needed()
                            time.sleep(0.5)
                            
                            # Get the href attribute if it's a link
                            href = None
                            try:
                                href = element.get_attribute("href")
                            except:
                                pass
                                
                            # Handle navigation (either expected or from href)
                            if expect_navigation or (href and not href.startswith(("#", "javascript:", "mailto:", "tel:"))):
                                success, message = handle_navigation(element, href)
                                return message
                            else:
                                # Regular click for non-navigation elements
                                # Try multiple click methods
                                try:
                                    self.page.evaluate("(element) => element.click()", element)
                                except:
                                    try:
                                        element.click()
                                    except:
                                        bbox = element.bounding_box()
                                        if bbox:
                                            self.page.mouse.click(
                                                bbox["x"] + bbox["width"] / 2,
                                                bbox["y"] + bbox["height"] / 2
                                            )
                                time.sleep(0.5)
                                return "Successfully clicked element"
                    except Exception as selector_error:
                        print(f"Specific selector failed: {str(selector_error)}")
                        # Fall through to try comprehensive selectors
                
                # Try comprehensive selectors if specific selector failed or wasn't provided
                for selector_type, selectors in click_selectors.items():
                    for base_selector in selectors:
                        try:
                            # Try exact text match
                            element = self.page.locator(f"{base_selector}:text-is('{target_text}')").first
                            if element and element.is_visible():
                                element.scroll_into_view_if_needed()
                                time.sleep(0.5)
                                
                                # Get href if it's a link
                                href = None
                                try:
                                    href = element.get_attribute("href")
                                except:
                                    pass
                                
                                # Handle navigation if needed
                                if expect_navigation or (href and not href.startswith(("#", "javascript:", "mailto:", "tel:"))):
                                    success, message = handle_navigation(element, href, selector_type)
                                    return message
                                else:
                                    # Try multiple click methods
                                    try:
                                        self.page.evaluate("(element) => element.click()", element)
                                    except:
                                        try:
                                            element.click()
                                        except:
                                            bbox = element.bounding_box()
                                            if bbox:
                                                self.page.mouse.click(
                                                    bbox["x"] + bbox["width"] / 2,
                                                    bbox["y"] + bbox["height"] / 2
                                                )
                                    time.sleep(1)
                                    return f"Clicked {selector_type} with text: {target_text}"
                                
                            # Try case-insensitive contains
                            element = self.page.locator(f"{base_selector}:text-matches('{target_text}', 'i')").first
                            if element and element.is_visible():
                                element.scroll_into_view_if_needed()
                                time.sleep(0.5)
                                
                                # Get href if it's a link
                                href = None
                                try:
                                    href = element.get_attribute("href")
                                except:
                                    pass
                                
                                # Handle navigation if needed
                                if expect_navigation or (href and not href.startswith(("#", "javascript:", "mailto:", "tel:"))):
                                    success, message = handle_navigation(element, href, selector_type)
                                    return message
                                else:
                                    # Try multiple click methods
                                    try:
                                        self.page.evaluate("(element) => element.click()", element)
                                    except:
                                        try:
                                            element.click()
                                        except:
                                            bbox = element.bounding_box()
                                            if bbox:
                                                self.page.mouse.click(
                                                    bbox["x"] + bbox["width"] / 2,
                                                    bbox["y"] + bbox["height"] / 2
                                                )
                                    time.sleep(1)
                                    return f"Clicked {selector_type} containing text: {target_text}"
                                
                            # Try aria-label and title attributes
                            element = self.page.locator(f"{base_selector}[aria-label*='{target_text}' i], {base_selector}[title*='{target_text}' i]").first
                            if element and element.is_visible():
                                element.scroll_into_view_if_needed()
                                time.sleep(0.5)
                                
                                # Get href if it's a link
                                href = None
                                try:
                                    href = element.get_attribute("href")
                                except:
                                    pass
                                
                                # Handle navigation if needed
                                if expect_navigation or (href and not href.startswith(("#", "javascript:", "mailto:", "tel:"))):
                                    success, message = handle_navigation(element, href, selector_type)
                                    return message
                                else:
                                    # Try multiple click methods
                                    try:
                                        self.page.evaluate("(element) => element.click()", element)
                                    except:
                                        try:
                                            element.click()
                                        except:
                                            bbox = element.bounding_box()
                                            if bbox:
                                                self.page.mouse.click(
                                                    bbox["x"] + bbox["width"] / 2,
                                                    bbox["y"] + bbox["height"] / 2
                                                )
                                    time.sleep(1)
                                    return f"Clicked {selector_type} with matching aria-label/title: {target_text}"
                        except:
                            continue
                
                return "Could not find clickable element with specified text"
                
            except Exception as e:
                return f"Error during click action: {str(e)}"
                
        elif action_type == "scroll":
            try:
                direction = action_data.get("text", "").lower()
                
                # Try JavaScript scrolling first
                try:
                    if direction == "top":
                        self.page.evaluate("window.scrollTo({ top: 0, behavior: 'smooth' })")
                    elif direction == "bottom":
                        self.page.evaluate("""
                            window.scrollTo({
                                top: document.body.scrollHeight,
                                behavior: 'smooth'
                            });
                        """)
                        time.sleep(1)
                        # Second scroll for dynamic content
                        self.page.evaluate("""
                            window.scrollTo({
                                top: document.body.scrollHeight,
                                behavior: 'smooth'
                            });
                        """)
                    else:
                        try:
                            amount = int(direction)
                        except:
                            amount = 500
                        self.page.evaluate(f"""
                            window.scrollBy({{
                                top: {amount},
                                behavior: 'smooth'
                            }});
                        """)
                except Exception as js_error:
                    print(f"JavaScript scroll failed: {str(js_error)}")
                    # Fallback to mouse wheel
                    if direction == "top":
                        self.page.mouse.wheel(0, -10000)
                    elif direction == "bottom":
                        self.page.mouse.wheel(0, 10000)
                        time.sleep(1)
                        self.page.mouse.wheel(0, 10000)
                    else:
                        try:
                            amount = int(direction)
                        except:
                            amount = 500
                        self.page.mouse.wheel(0, amount)
                
                time.sleep(1)
                return f"Scrolled {direction}"
                
            except Exception as e:
                return f"Error scrolling: {str(e)}"
                
        elif action_type == "navigate":
            try:
                url = action_data.get("text", "")
                if not url.startswith(("http://", "https://")):
                    url = "https://" + url
                
                # Enhanced navigation with retries
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        if attempt == 0:
                            self.page.goto(url, wait_until="networkidle", timeout=30000)
                        elif attempt == 1:
                            self.page.goto(url, wait_until="domcontentloaded", timeout=30000)
                        else:
                            self.page.goto(url, wait_until="load", timeout=30000)
                        break
                    except Exception as nav_error:
                        print(f"Navigation attempt {attempt + 1} failed: {str(nav_error)}")
                        if attempt == max_retries - 1:
                            raise nav_error
                        time.sleep(2)
                
                # Wait for page stability
                try:
                    self.page.wait_for_selector("body", timeout=5000)
                    time.sleep(2)
                except:
                    pass
                
                return f"Navigated to {url}"
                
            except Exception as e:
                return f"Error navigating: {str(e)}"
        
        else:
            return f"Unknown action type: {action_type}"
    
    def _get_fallback_action(self, task: str) -> Tuple[Dict[str, Any], str]:
        """Get a fallback action based on the task description."""
        tl = task.lower()
        
        # Extract full email address with special handling for + and other special characters
        email_pattern = r'[\w\.-]+(?:\+[\w\.-]+)?@[\w\.-]+\.\w+'
        email_match = re.search(email_pattern, task)
        email = email_match.group(0) if email_match else None
        
        # Extract password (anything after "password" until end or separator)
        password = None
        if "password" in tl:
            # Find the index in the original task to preserve case
            password_idx = task.lower().index("password") + 8
            password = task[password_idx:].strip().strip('"').strip("'")
            # Remove any trailing separators or punctuation
            for sep in [' then ', ' and ', '.', ',', ';']:
                if password.endswith(sep):
                    password = password.rsplit(sep, 1)[0].strip()
        
        # Adobe-specific selectors
        adobe_selectors = {
            "sign_in": [
                # Primary sign-in buttons
                "[data-testid='Sign in']",
                "[data-testid='sign-in']",
                "[data-testid='login']",
                "[data-testid='signin']",
                "[data-testid='sign-in-button']",
                "[data-testid='login-button']",
                
                # Adobe-specific button classes
                "button.spectrum-Button--cta",
                "button.spectrum-Button--primary",
                "button.spectrum-Button--overBackground",
                
                # Text-based selectors (case-insensitive)
                "button:text-is('Sign in')",
                "button:text-is('sign in')",
                "button:text-is('Sign In')",
                "button:text-is('SIGN IN')",
                "a:text-is('Sign in')",
                "a:text-is('sign in')",
                "a:text-is('Sign In')",
                ".spectrum-Button:text-is('Sign in')",
                ".spectrum-Button:text-is('sign in')",
                ".spectrum-Button:text-is('Sign In')",
                
                # Role-based selectors
                "[role='button']:text-is('Sign in')",
                "[role='button']:text-is('sign in')",
                "[role='button']:text-is('Sign In')",
                
                # Link-based selectors
                "a[href*='signin']",
                "a[href*='login']",
                
                # Fallback selectors
                "button.spectrum-Button",
                ".spectrum-Button"
            ],
            "email": [
                # Primary email selectors
                "input[type='email']",
                "input[name='email']",
                "input[id*='email']",
                "input[placeholder*='email']",
                "input[autocomplete='email']",
                "[data-testid*='email']",
                
                # Adobe-specific email selectors
                "[data-testid='EmailPage-email']",
                "[data-testid='EmailField']",
                "[data-testid='EmailInput']",
                ".spectrum-Textfield[name*='email']",
                
                # Generic text inputs that might be email
                "input.spectrum-Textfield",
                "input[type='text']",
                ".spectrum-Textfield"
            ],
            "password": [
                # Primary password selectors
                "input[type='password']",
                "input[name='password']",
                "input[id*='password']",
                "input[autocomplete='current-password']",
                "[data-testid*='password']",
                
                # Adobe-specific password selectors
                "[data-testid='PasswordPage-password']",
                "[data-testid='Password']",
                "[data-testid='PasswordField']",
                "[data-testid='PasswordInput']",
                "[data-testid*='password-input']",
                ".spectrum-Textfield[type='password']",
                ".spectrum-Textfield[name*='password']",
                
                # Generic password inputs
                "input.spectrum-Textfield[type='password']",
                ".spectrum-Textfield input[type='password']",
                "form input[type='password']",
                "#password",
                ".password-input"
            ],
            "continue": [
                # Primary Adobe continue button selectors
                "[data-testid='EmailPage-submitButton']",
                "[data-testid='ContinueButton']",
                "[data-testid='continue-button']",
                "[data-testid='submit-button']",
                "button[data-testid*='continue']",
                "button[data-testid*='submit']",
                
                # Adobe-specific button styles
                "button.spectrum-Button--overBackground:not([disabled])",
                "button.spectrum-Button--fill:not([disabled])",
                "button.spectrum-Button--cta:not([disabled])",
                "button.spectrum-Button--primary:not([disabled])",
                
                # Form submit buttons
                "button[type='submit']:not([disabled])",
                "input[type='submit']:not([disabled])",
                
                # Text-based selectors
                "button:text-is('Continue'):not([disabled])",
                "button:text-is('continue'):not([disabled])",
                "button:text-is('Next'):not([disabled])",
                "button:text-is('next'):not([disabled])",
                ".spectrum-Button:text-is('Continue'):not([disabled])",
                ".spectrum-Button:text-is('continue'):not([disabled])",
                ".spectrum-Button:text-is('Next'):not([disabled])",
                ".spectrum-Button:text-is('next'):not([disabled])",
                
                # Generic enabled buttons
                "button.spectrum-Button:not([disabled]):not([aria-disabled='true'])",
                ".spectrum-Button:not([disabled]):not([aria-disabled='true'])"
            ]
        }
        
        # Handle wait tasks
        if "wait" in tl:
            if "page" in tl:
                return {
                    "action": "wait",
                    "text": "2"
                }, "fallback: wait for page load"
            elif "email" in tl:
                return {
                    "action": "wait",
                    "selector": ", ".join(adobe_selectors["email"])
                }, "fallback: wait for email field"
            elif "password" in tl:
                return {
                    "action": "wait",
                    "selector": ", ".join(adobe_selectors["password"])
                }, "fallback: wait for password field"
            elif "button" in tl:
                return {
                    "action": "wait",
                    "selector": ", ".join(adobe_selectors["continue"])
                }, "fallback: wait for button"
            elif "validation" in tl:
                return {
                    "action": "wait",
                    "text": "1"
                }, "fallback: wait for validation"
            else:
                return {
                    "action": "wait",
                    "text": "1"
                }, "fallback: general wait"
        
        # Handle sign-in related tasks
        if "sign in" in tl or "login" in tl:
            return {
                "action": "click",
                "selector": ", ".join(adobe_selectors["sign_in"]),
                "text": "sign in"
            }, "fallback: click Adobe sign in button"
        elif "email" in tl and email:
            return {
                "action": "type",
                "selector": ", ".join(adobe_selectors["email"]),
                "text": email
            }, f"fallback: type email {email}"
        elif "password" in tl and password:
            return {
                "action": "type",
                "selector": ", ".join(adobe_selectors["password"]),
                "text": password
            }, "fallback: type password"
        elif "continue" in tl or "next" in tl:
            return {
                "action": "click",
                "selector": ", ".join(adobe_selectors["continue"]),
                "text": "continue"
            }, "fallback: click continue button"

        # Original fallback actions for other cases
        search_terms = None
        if "type" in tl:
            # Extract text between "type" and "and" or end of string
            type_idx = tl.index("type") + 4
            and_idx = tl.find(" and ", type_idx)
            if and_idx != -1:
                search_terms = tl[type_idx:and_idx].strip()
            else:
                search_terms = tl[type_idx:].strip()
        
        # Prioritized fallback actions
        if task.startswith(("http://", "https://")):
            return {"action": "navigate", "text": task}, "fallback: direct URL navigation"
        elif search_terms:
            return {
                "action": "type",
                "selector": "input[type='search']",
                "text": search_terms
            }, f"fallback: type action with text: {search_terms}"
        elif "scroll" in tl and "bottom" in tl:
            return {"action": "scroll", "text": "bottom"}, "fallback: scroll to bottom"
        elif "submit" in tl:
            return {
                "action": "click",
                "selector": "button[type='submit'], input[type='submit']",
                "text": "submit"
            }, "fallback: submit action"
        elif "wait" in tl:
            return {"action": "wait", "text": "2"}, "fallback: wait action"
        else:
            # Default to searching if no other action matches
            return {
                "action": "type",
                "selector": "input[type='search']",
                "text": task
            }, "fallback: default type action"
    
    def parse_sequential_tasks(self, task_description: str) -> List[str]:
        """Parse a complex task description into sequential subtasks."""
        # Task keywords and their variations
        task_patterns = {
            'login': ['login', 'sign in', 'signin'],
            'type': ['type', 'enter', 'input', 'fill'],
            'click': ['click', 'press', 'select', 'choose'],
            'scroll': ['scroll', 'move'],
            'navigate': ['go to', 'visit', 'open', 'navigate to'],
            'search': ['search', 'look for', 'find']
        }
        
        # Common task separators
        separators = [
            ' then ',
            ' and then ',
            ' after that ',
            ', then ',
            '; ',
            ' next ',
            ' followed by ',
            ' when ',
            ' after ',
            ' once '
        ]
        
        # Split tasks based on separators
        tasks = [task_description]
        for separator in separators:
            new_tasks = []
            for task in tasks:
                split_tasks = task.split(separator)
                new_tasks.extend([t.strip() for t in split_tasks if t.strip()])
            tasks = new_tasks
        
        # Process tasks and extract login sequences
        processed_tasks = []
        i = 0
        while i < len(tasks):
            task = tasks[i]  # Keep original case
            task_lower = task.lower()  # Use lowercase only for comparison
            
            # Handle login sequences
            if any(login_word in task_lower for login_word in task_patterns['login']):
                # Look ahead for email/username and password
                login_sequence = []
                for j in range(i, min(i + 3, len(tasks))):
                    curr_task = tasks[j]  # Keep original case
                    curr_task_lower = curr_task.lower()  # Use lowercase only for comparison
                    if '@' in curr_task or 'email' in curr_task_lower or 'username' in curr_task_lower:
                        login_sequence.append(f"type {curr_task}")  # Use original case
                    elif 'password' in curr_task_lower:
                        login_sequence.append(f"type {curr_task}")  # Use original case
                        if j + 1 < len(tasks) and any(click in tasks[j + 1].lower() for click in task_patterns['click']):
                            login_sequence.append(tasks[j + 1])  # Use original case
                            i = j + 2
                        else:
                            login_sequence.append("click login")
                            i = j + 1
                        break
                if login_sequence:
                    processed_tasks.extend(login_sequence)
                else:
                    processed_tasks.append(task)  # Use original case
                    i += 1
            
            # Handle search sequences
            elif any(search_word in task_lower for search_word in task_patterns['search']):
                search_text = task.split(next(word for word in task_patterns['search'] if word.lower() in task_lower))[1].strip()
                processed_tasks.append(f"type {search_text}")  # Use original case
                processed_tasks.append("press Enter")
                i += 1
            
            # Handle scroll commands
            elif any(scroll_word in task_lower for scroll_word in task_patterns['scroll']):
                processed_tasks.append("wait for page load")
                processed_tasks.append(task)  # Use original case
                processed_tasks.append("wait for page load")
                i += 1
            
            # Handle navigation
            elif any(nav_word in task_lower for nav_word in task_patterns['navigate']):
                url = task.split(next(word for word in task_patterns['navigate'] if word.lower() in task_lower))[1].strip()
                processed_tasks.append(f"navigate {url}")  # Use original case
                processed_tasks.append("wait for page load")
                i += 1
            
            # Handle other tasks
            else:
                processed_tasks.append(task)  # Use original case
                i += 1
        
        print("\nParsed tasks:")
        for i, task in enumerate(processed_tasks, 1):
            print(f"{i}. {task}")
                
        return processed_tasks
    
    def run_task(self, initial_url: str, task: str, max_steps: int = 10) -> str:
        """Run a task starting from the given URL."""
        try:
            print(f"\nStarting task sequence: {task}")
            print(f"Navigating to: {initial_url}")
            
            # Parse the task into sequential subtasks
            subtasks = self.parse_sequential_tasks(task)
            total_steps = max_steps
            steps_per_task = max(total_steps // len(subtasks), 3)  # Ensure at least 3 steps per subtask
            
            # Enhanced navigation with retries and fallbacks
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    # First try with networkidle
                    if attempt == 0:
                        print("Attempting navigation with networkidle...")
                        self.page.goto(initial_url, wait_until="networkidle", timeout=30000)
                    # Second try with domcontentloaded
                    elif attempt == 1:
                        print("Retrying with domcontentloaded...")
                        self.page.goto(initial_url, wait_until="domcontentloaded", timeout=30000)
                    # Last try with load
                    else:
                        print("Final attempt with load...")
                        self.page.goto(initial_url, wait_until="load", timeout=30000)
                    
                    print("✓ Page loaded successfully")
                    break
                except Exception as nav_error:
                    print(f"Navigation attempt {attempt + 1} failed: {str(nav_error)}")
                    if attempt == max_retries - 1:
                        return f"Failed to load page after {max_retries} attempts: {str(nav_error)}"
                    time.sleep(2)  # Wait before retrying
            
            # Additional wait for page stability
            try:
                print("Waiting for page to stabilize...")
                self.page.wait_for_selector("body", timeout=5000)
                time.sleep(2)  # Short additional wait for dynamic content
            except Exception as e:
                print(f"Warning: Additional stability wait failed: {str(e)}")
            
            # Execute each subtask
            results = []
            for subtask_index, subtask in enumerate(subtasks, 1):
                print(f"\nExecuting subtask {subtask_index}/{len(subtasks)}: {subtask}")
                
                # Reset counters for this subtask
                step = 0
                last_action = None
                consecutive_same_actions = 0
                
                # Execute steps for this subtask
                while step < steps_per_task:
                    step += 1
                    print(f"\nSubtask {subtask_index} - Step {step}/{steps_per_task}")
                    
                    try:
                        # Get current page state
                        print("Getting page state...")
                        page_state = self.get_page_state()
                        print("✓ Got page state")
                        
                        # Analyze page and decide next action
                        print(f"Analyzing page for subtask: {subtask}")
                        action, reasoning = self.analyze_page_and_decide(subtask)
                        print(f"✓ Decided action: {json.dumps(action, indent=2)}")
                        print(f"Reasoning: {reasoning[:200]}...")
                        
                        # If no valid action, try next subtask
                        if action is None:
                            results.append(f"Subtask {subtask_index} - Step {step}: Could not determine next action.")
                            break
                        
                        # Execute the action
                        print(f"Executing action: {action['action']}")
                        action_json_str = json.dumps(action)
                        result = self.execute_action(action_json_str)
                        results.append(f"Subtask {subtask_index} - Step {step}: {result}")
                        print(f"Action result: {result}")
                        
                        # Check for successful search completion
                        if (action['action'] == 'type' and 
                            'search' in result.lower() and 
                            'submitted' in result.lower() and 
                            not 'error' in result.lower()):
                            print("✓ Search task completed successfully")
                            results.append(f"Subtask {subtask_index} completed successfully!")
                            break
                            
                        # Check for other successful actions
                        if ('successfully' in result.lower() and 
                            not 'error' in result.lower() and
                            (
                                # For type actions (non-search), check if we typed what was requested
                                (action['action'] == 'type' and 
                                 action.get('text', '').lower() in result.lower()) or
                                # For click actions, check if we clicked what was requested
                                (action['action'] == 'click' and 
                                 action.get('text', '').lower() in result.lower()) or
                                # For navigation, check if we reached the URL
                                (action['action'] == 'navigate' and 
                                 'navigated to' in result.lower()) or
                                # For scroll actions, check if we scrolled as requested
                                (action['action'] == 'scroll' and 
                                 'scrolled' in result.lower())
                            )):
                            print(f"✓ Action completed successfully: {action['action']}")
                            
                            # Check if this was the last required action for the subtask
                            if (
                                # If this was a navigation task and we navigated
                                ('go to' in subtask.lower() and action['action'] == 'navigate') or
                                # If this was a click task and we clicked
                                ('click' in subtask.lower() and action['action'] == 'click') or
                                # If this was a type task (non-search) and we typed
                                ('type' in subtask.lower() and 'search' not in subtask.lower() and action['action'] == 'type') or
                                # If this was a scroll task and we scrolled
                                ('scroll' in subtask.lower() and action['action'] == 'scroll')
                            ):
                                results.append(f"Subtask {subtask_index} completed successfully!")
                                break

                        # Check for task completion from AI reasoning
                        if "task complete" in reasoning.lower():
                            results.append(f"Subtask {subtask_index} completed successfully!")
                            break
                        
                        # Check for stuck state
                        if last_action == action:
                            consecutive_same_actions += 1
                            if consecutive_same_actions >= 3:
                                results.append(f"Subtask {subtask_index}: Detected potential stuck state. Moving to next subtask.")
                                break
                        else:
                            consecutive_same_actions = 0
                            last_action = action
                        
                        # Wait for page to load/react with reduced timeout
                        try:
                            print("Waiting for page to stabilize...")
                            self.page.wait_for_load_state("domcontentloaded", timeout=5000)
                            print("✓ Page stable")
                        except Exception as e:
                            print(f"Warning: Page load wait timed out: {str(e)}")
                        
                    except Exception as step_error:
                        print(f"Error in subtask {subtask_index} step {step}: {str(step_error)}")
                        results.append(f"Subtask {subtask_index} - Step {step}: Error: {str(step_error)}")
                        continue
                
            return "\n".join(results)
                
        except Exception as e:
            return f"Task execution failed: {str(e)}"
    
    def cleanup(self):
        """Clean up browser resources."""
        try:
            if self.page:
                self.page.close()
                self.page = None
            if self.context:
                self.context.close()
                self.context = None
            if self.browser:
                self.browser.close()
                self.browser = None
            if self.playwright:
                self.playwright.stop()
                self.playwright = None
        except Exception as e:
            print(f"Cleanup error: {str(e)}")

# Define the Gradio interface class
class WebAgentGradioInterface:
    """A Gradio interface for the web agent with visual perception."""
    
    def __init__(self):
        """Initialize the web agent Gradio interface."""
        self.agent = None
        self.is_running = False
        self.debug_mode = True
        self.current_task = None
        self.progress = 0
        self.status_message = ""

    def create_interface(self):
        """Create and return the Gradio interface."""
        with gr.Blocks(title="Web Agent with Visual Perception") as interface:
            gr.Markdown("# Web Agent with Visual Perception")
            gr.Markdown("Enter a URL and task description to automate web interactions using visual AI.")
            
            with gr.Row():
                with gr.Column(scale=2):
                    url_input = gr.Textbox(
                        label="URL",
                        placeholder="www.example.com",
                        info="Enter the starting URL for the web agent"
                    )
                    
                    task_input = gr.Textbox(
                        label="Task Description",
                        placeholder="Example: Login with email user@example.com, navigate to dashboard, and download the latest report",
                        info="Describe the task you want the agent to perform",
                        lines=3
                    )
                    
                    with gr.Row():
                        max_steps = gr.Slider(
                            minimum=1,
                            maximum=20,
                            value=10,
                            step=1,
                            label="Maximum Steps",
                            info="Maximum number of actions the agent will take"
                        )
                        
                        headless = gr.Checkbox(
                            label="Run Headless",
                            value=False,
                            info="Run browser in headless mode (no visible window)"
                        )
                    
                    run_button = gr.Button("Run Web Agent", variant="primary")
                    stop_button = gr.Button("Stop", variant="stop")
                    
                with gr.Column(scale=3):
                    status = gr.Markdown("Status: Ready")
                    
                    with gr.Tabs():
                        with gr.TabItem("Results"):
                            result_text = gr.Textbox(
                                label="Action Log",
                                lines=10,
                                max_lines=20,
                                show_copy_button=True
                            )
                            
                            result_gallery = gr.Gallery(
                                label="Screenshots",
                                columns=2,
                                height="500px",
                                show_download_button=True,
                                elem_id="result_gallery"
                            )
                            
                        with gr.TabItem("Debug Info"):
                            debug_text = gr.Textbox(
                                label="Debug Log",
                                lines=20,
                                visible=self.debug_mode
                            )
            
            def stop_agent():
                if self.is_running:
                    self.is_running = False
                    return gr.update(value="Status: Stopped by user")
            
            # Set up event handlers
            run_button.click(
                fn=self.run_agent_with_progress,
                inputs=[url_input, task_input, max_steps, headless],
                outputs=[result_text, result_gallery, status]
            )
            
            stop_button.click(
                fn=stop_agent,
                outputs=[status]
            )
            
        return interface

    def run_agent_with_progress(self, url, task, max_steps, headless):
        """Run the agent with progress updates."""
        self.is_running = True
        self.current_task = task
        self.progress = 0
        result_text = ""
        screenshots = []
        
        try:
            # Initialize progress
            print("Setting up agent...")
            
            # Setup agent
            if not self.agent:
                self.agent = WebPerceptionAgent(headless=headless)
                if not self.agent.setup():
                    raise Exception("Failed to set up web agent")
            
            # Run the task
            print("Starting task...")
            
            # Ensure URL starts with http:// or https://
            if not url.startswith(("http://", "https://")):
                url = "https://" + url
            
            # Run the task and get results
            result_text = self.agent.run_task(url, task, max_steps)
            
            # Get final screenshot
            final_screenshot = self.agent.take_screenshot("Task complete")
            if final_screenshot:
                screenshots.append(final_screenshot['image'])
            
            return result_text, screenshots, gr.update(value="Status: Task completed")
            
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            print(error_msg)
            return error_msg, screenshots, gr.update(value="Status: Error occurred")
        finally:
            self.is_running = False

    def cleanup(self):
        """Clean up resources."""
        if self.agent:
            self.agent.cleanup()
            self.agent = None
        self.is_running = False

    def launch(self, **kwargs):
        """Launch the Gradio interface."""
        interface = self.create_interface()
        interface.launch(**kwargs)

def check_system_requirements():
    """Check if all system requirements are met."""
    try:
        # Check Together.ai configuration
        if not TOGETHER_API_KEY or TOGETHER_API_KEY == "YOUR_API_KEY_HERE":
            print("✗ Together.ai API key not configured")
            print("Update your API key in the TOGETHER_API_KEY variable")
            return False
        
        # Test Together.ai connection
        try:
            client = TogetherClient(api_key=TOGETHER_API_KEY)
            response = client.chat_completions(messages=[{"role": "user", "content": "Hello"}])
            print("✓ Together.ai connection successful")
        except Exception as e:
            print(f"✗ Together.ai setup failed: {str(e)}")
            print("Check your API key and internet connection")
            return False
            
        # Check Gradio
        try:
            import gradio as gr
            print("✓ Gradio is installed")
        except ImportError:
            print("✗ Gradio not installed")
            print("Try running: pip install gradio")
            return False
        
        print("\n✓ All system requirements met")
        return True
        
    except Exception as e:
        print(f"✗ System check failed: {str(e)}")
        print("Check your API key and internet connection")
        return False

def main():
    """Main function to run the application."""
    try:
        print("\n=== Starting Web Agent Application ===")
        
        # Check system requirements
        if not check_system_requirements():
            print("\nPlease fix the above issues and try again.")
            return
        
        print("\nStarting Gradio interface...")
        app = WebAgentGradioInterface()
        
        # Launch with error handling
        try:
            app.launch(share=False, inbrowser=True)
        except Exception as e:
            print(f"\nError launching Gradio interface: {str(e)}")
            print("\nTroubleshooting steps:")
            print("1. Check if port 7860 is available")
            print("2. Ensure you have sufficient permissions")
            print("3. Try restarting your computer")
            print("4. Check your firewall settings")
        
    except ImportError as e:
        print(f"\nError: Missing required package - {str(e)}")
        print("Please run: pip install -r requirements.txt")
    except Exception as e:
        print(f"\nError starting the application: {str(e)}")
        print("\nTroubleshooting steps:")
        print("1. Make sure all requirements are installed: pip install -r requirements.txt")
        print("2. Install Playwright browsers: playwright install")
        print("3. Configure your Together.ai API key in the TOGETHER_API_KEY variable")
        print("4. Check your internet connection")
        print("5. Check system resources (CPU, memory)")
        print("6. Try restarting your computer")

if __name__ == "__main__":
    main()
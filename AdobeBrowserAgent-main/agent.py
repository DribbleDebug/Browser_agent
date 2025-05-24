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
from io import BytesIO
from typing import Dict, Any, Optional, Tuple, List
import re

from playwright.sync_api import sync_playwright
import gradio as gr
from PIL import Image
from together import Together

import config

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
    
    def __init__(self, headless: bool = config.DEFAULT_HEADLESS):
        """Initialize the web perception agent.
        
        Args:
            headless: Whether to run the browser in headless mode (no GUI)
        
        The agent initializes:
        - Browser automation components (Playwright)
        - AI components (Together.ai client)
        - State tracking (history, screenshots, status)
        """
        # Browser automation settings
        self.headless = headless
        self.browser = None
        self.page = None
        self.context = None
        self.playwright = None
        
        # Initialize AI client
        try:
            self.together_client = Together(api_key=config.TOGETHER_API_KEY)
            if not self.together_client:
                raise Exception("Failed to initialize Together client")
        except Exception as e:
            print(f"Warning: Together client initialization failed: {str(e)}")
            self.together_client = None
            
        # State tracking
        self.last_action = None  # Most recent action executed
        self.action_history = []  # History of all actions
        self.screenshots = []  # Screenshots with metadata
        self.current_status = ""  # Current agent status
        
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
        """Analyze the current page and decide on the next action.
        
        This is the core decision-making method that:
        1. Analyzes the current page state
        2. Interprets the user's task
        3. Decides what action to take next
        
        The method handles common scenarios:
        - Navigation to URLs
        - Search/typing tasks
        - Clicking elements
        - Scrolling
        - Fallback to AI model for complex decisions
        
        Args:
            task: The task description from the user
            
        Returns:
            Tuple containing:
            - Dict with the next action to execute
            - String explaining the reasoning
        """
        try:
            # Get current page state
            state = self.get_page_state()
            task_lower = task.lower()
            
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

            # Handle search tasks using generic selectors
            if any(word in task_lower for word in ["search", "find", "look up", "type"]):
                # Extract search terms after action word
                search_terms = None
                for word in ["search", "find", "look up", "type"]:
                    if word in task_lower:
                        search_terms = task_lower.split(word, 1)[1].strip()
                        break
                
                if search_terms:
                    return {
                        "action": "type",
                        "selector": """
                            input[type='search'],
                            input[name='q'],
                            input[name='query'],
                            input[name='search'],
                            input[placeholder*='search' i],
                            .search-input,
                            #search-input,
                            input[type='text']
                        """,
                        "text": search_terms
                    }, "Typing search terms"

            # Handle click tasks with position or text targeting
            if "click" in task_lower:
                # Extract text after "click"
                click_text = task_lower.split("click", 1)[1].strip()
                
                # Handle numerical positions (e.g., "click 3rd link")
                position_match = re.search(r'(\d+)(?:st|nd|rd|th)?\s+(?:link|button)', click_text)
                if position_match:
                    position = int(position_match.group(1))
                    return {
                        "action": "click",
                        "selector": f"a[href]:not([href='']):not([href^='#']):nth-child({position}), a[href]:not([href='']):not([href^='#']):nth-of-type({position})"
                    }, f"Clicking link number {position}"
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
                            button:contains('Login'), 
                            button:contains('Sign in'),
                            a:contains('Login'),
                            a:contains('Sign in'),
                            [role='button']:contains('Login'),
                            [role='button']:contains('Sign in'),
                            .login-button,
                            .signin-button,
                            #login,
                            #signin
                        """
                    }, "Clicking login button"
                elif "submit" in click_text:
                    return {
                        "action": "click",
                        "selector": "button[type='submit'], input[type='submit'], button:contains('Submit'), .submit-button, #submit"
                    }, "Clicking submit button"
                else:
                    # Generic click by text content
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
                    self.together_client = Together(api_key=config.TOGETHER_API_KEY)
                
                response = self.together_client.chat.completions.create(
                    messages=messages,
                    model=config.DEFAULT_CHAT_MODEL,
                    temperature=0.2,
                    max_tokens=config.DEFAULT_MAX_TOKENS
                )
                
                if response and hasattr(response, 'choices') and len(response.choices) > 0:
                    content = response.choices[0].message.content.strip()
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
                    self.together_client = Together(api_key=config.TOGETHER_API_KEY)
                
                # Test with a simple chat completion
                response = self.together_client.chat.completions.create(
                    messages=[{
                        "role": "user",
                        "content": "Hello"
                    }],
                    model=config.DEFAULT_CHAT_MODEL,
                    max_tokens=10,
                    temperature=0.7
                )
                
                if response and hasattr(response, 'choices') and len(response.choices) > 0:
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
        
        if action_type == "navigate":
            url = action_data.get("text", "")
            if not url.startswith(("http://", "https://")):
                url = "https://" + url
                
            # Enhanced page load handling
            try:
                # First try with networkidle
                self.page.goto(url, wait_until="networkidle", timeout=30000)
            except Exception as e:
                print(f"Network idle timeout, falling back to domcontentloaded: {str(e)}")
                try:
                    # Fallback to domcontentloaded
                    self.page.goto(url, wait_until="domcontentloaded", timeout=30000)
                except Exception as e2:
                    print(f"DOM content load timeout, using basic load: {str(e2)}")
                    # Final fallback
                    self.page.goto(url, wait_until="load", timeout=30000)
            
            # Additional wait for stability
            try:
                self.page.wait_for_selector("body", timeout=5000)
                time.sleep(2)  # Short wait for dynamic content
            except:
                pass
                
            return f"Navigated to {url}"
            
        elif action_type == "scroll":
            direction = action_data.get("text", "").lower()
            try:
                # First try using JavaScript scrolling
                try:
                    if direction == "top":
                        self.page.evaluate("window.scrollTo({ top: 0, behavior: 'smooth' })")
                    elif direction == "bottom" or "bottom" in direction:
                        # First scroll attempt
                        self.page.evaluate("""
                            window.scrollTo({
                                top: document.body.scrollHeight,
                                behavior: 'smooth'
                            })
                        """)
                        time.sleep(1)  # Wait for dynamic content
                        
                        # Second scroll to handle any dynamically loaded content
                        self.page.evaluate("""
                            window.scrollTo({
                                top: document.body.scrollHeight,
                                behavior: 'smooth'
                            })
                        """)
                    else:
                        try:
                            # Try to parse as number of pixels
                            amount = int(direction)
                            self.page.evaluate(f"""
                                window.scrollBy({{
                                    top: {amount},
                                    behavior: 'smooth'
                                }})
                            """)
                        except ValueError:
                            # Default scroll amount (one viewport height)
                            self.page.evaluate("""
                                window.scrollBy({
                                    top: window.innerHeight * 0.8,
                                    behavior: 'smooth'
                                })
                            """)
                    
                    time.sleep(0.5)  # Wait for smooth scroll to complete
                    return f"Scrolled {direction}"
                    
                except Exception as js_error:
                    print(f"JavaScript scroll failed: {str(js_error)}")
                    # Fallback to Playwright's mouse wheel simulation
                    try:
                        if direction == "top":
                            self.page.mouse.wheel(0, -10000)
                        elif direction == "bottom" or "bottom" in direction:
                            self.page.mouse.wheel(0, 10000)
                            time.sleep(1)
                            self.page.mouse.wheel(0, 10000)  # Second attempt for dynamic content
                        else:
                            try:
                                amount = int(direction)
                                self.page.mouse.wheel(0, amount)
                            except ValueError:
                                self.page.mouse.wheel(0, 500)  # Default amount
                        
                        time.sleep(0.5)
                        return f"Scrolled {direction} using mouse wheel"
                    except Exception as wheel_error:
                        print(f"Mouse wheel scroll failed: {str(wheel_error)}")
                        # Final fallback: try keyboard
                        if direction == "top":
                            self.page.keyboard.press("Home")
                        elif direction == "bottom" or "bottom" in direction:
                            self.page.keyboard.press("End")
                        else:
                            self.page.keyboard.press("PageDown")
                        return f"Scrolled {direction} using keyboard"
                        
            except Exception as e:
                return f"Error scrolling: {str(e)}"
            
        elif action_type == "click":
            selector = action_data.get("selector", "")
            try:
                # Wait for element with retry
                max_retries = 3
                retry_delay = 2
                for attempt in range(max_retries):
                    try:
                        element = self.page.wait_for_selector(selector, state="visible", timeout=10000)
                        if element:
                            element.click()
                            return f"Clicked element: {selector}"
                    except Exception as e:
                        if attempt < max_retries - 1:
                            print(f"Click attempt {attempt + 1} failed, retrying in {retry_delay} seconds...")
                            time.sleep(retry_delay)
                            continue
                        raise e
                        
                return f"Failed to click element: {selector}"
            except Exception as e:
                return f"Error clicking element: {str(e)}"
            
        elif action_type == "type":
            text = action_data.get("text", "")
            selector = action_data.get("selector", "")
            
            # Extract email and password from text if present
            email_pattern = r'[\w\.-]+(?:\+[\w\.-]+)?@[\w\.-]+\.\w+'
            email_match = re.search(email_pattern, text)
            
            # If this is an email field and we found an email in the text
            if (any(email_indicator in selector.lower() for email_indicator in ["email", "username", "login"]) and 
                email_match):
                text = email_match.group(0)
            # If this is a password field
            elif any(pwd_indicator in selector.lower() for pwd_indicator in ["password", "pwd"]):
                # Extract password (anything after "password" keyword)
                if "password" in text.lower():
                    pwd_start = text.lower().find("password") + 8
                    text = text[pwd_start:].strip().strip('"\'').split()[0]
            
            # If no selector provided, try to find a suitable input
            if not selector:
                # Common input selectors
                search_selectors = [
                    # Email/Username fields
                    "input[type='email']",
                    "input[name='email']",
                    "input[id*='email']",
                    "input[name='username']",
                    "input[id*='username']",
                    # Password fields
                    "input[type='password']",
                    "input[name='password']",
                    "input[id*='password']",
                    # Search fields
                    "input[type='search']",
                    "input[name='q']",
                    # Generic text inputs
                    "input[type='text']"
                ]
                
                # Try each selector with retry
                max_retries = 3
                retry_delay = 2
                
                for sel in search_selectors:
                    for attempt in range(max_retries):
                        try:
                            print(f"Trying selector: {sel} (attempt {attempt + 1})")
                            element = self.page.wait_for_selector(sel, state="visible", timeout=5000)
                            if element:
                                selector = sel
                                print(f"Found visible input with selector: {sel}")
                                break
                        except Exception as e:
                            if attempt < max_retries - 1:
                                print(f"Attempt {attempt + 1} failed, retrying in {retry_delay} seconds...")
                                time.sleep(retry_delay)
                                continue
                            print(f"Selector {sel} failed all attempts: {str(e)}")
                        
                    if selector:  # If we found a working selector, stop trying others
                        break
                
                if not selector:
                    return "Could not find a suitable input field after all attempts"
            
            # Wait for the element and type
            try:
                print(f"Using selector for typing: {selector}")
                element = self.page.wait_for_selector(selector, state="visible", timeout=10000)
                if not element:
                    return f"Could not find element with selector: {selector}"
                
                # Clear existing text
                element.fill("")
                time.sleep(0.5)  # Wait after clearing
                
                # Type the text
                element.fill(text)
                time.sleep(0.5)  # Wait after typing
                
                # Press Enter for search inputs
                if any(s in selector.lower() for s in ["search", "q", "query"]):
                    element.press("Enter")
                    time.sleep(1)  # Wait after submit
                    
                    # Wait for navigation if it happens
                    try:
                        # First wait for navigation to start
                        self.page.wait_for_load_state("domcontentloaded", timeout=5000)
                        
                        # Then wait for network idle (but don't fail if it times out)
                        try:
                            self.page.wait_for_load_state("networkidle", timeout=5000)
                        except:
                            pass
                        
                        # Additional wait for search results to load
                        time.sleep(1)
                        
                        # Check if we're on a search results page
                        if "search" in self.page.url.lower() or "wiki" in self.page.url.lower():
                            return f"Typed '{text}' into search and submitted"
                        else:
                            return f"Warning: Search submitted but unexpected URL: {self.page.url}"
                    except Exception as e:
                        return f"Warning: Search submitted but navigation failed: {str(e)}"
                
                return f"Typed '{text}' into {selector}"
            except Exception as e:
                return f"Error typing text: {str(e)}"
            
        elif action_type == "wait":
            if "selector" in action_data:
                selector = action_data["selector"]
                try:
                    self.page.wait_for_selector(selector, timeout=10000)
                    return f"Waited for element: {selector}"
                except Exception as e:
                    return f"Error waiting for element: {str(e)}"
            else:
                seconds = float(action_data.get("text", "1"))
                time.sleep(seconds)
                return f"Waited for {seconds} seconds"
        
        else:
            raise ValueError(f"Unknown action type: {action_type}")
    
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
        # Common task separators
        separators = [
            " then ",
            " and then ",
            " after that ",
            ", then ",
            "; ",
            " next ",
            " followed by ",
            " when ",
            " after ",
            " once "
        ]
        
        # Split tasks based on separators
        tasks = [task_description]
        for separator in separators:
            new_tasks = []
            for task in tasks:
                split_tasks = task.split(separator)
                new_tasks.extend([t.strip() for t in split_tasks if t.strip()])
            tasks = new_tasks
        
        # Process tasks and handle scroll commands specifically
        processed_tasks = []
        for task in tasks:
            task_lower = task.lower()
            
            # Handle scroll tasks
            if "scroll" in task_lower:
                # Add a wait before scrolling
                processed_tasks.append("wait for page load")
                # Add the scroll task directly
                processed_tasks.append(task)
                # Add a wait after scrolling
                processed_tasks.append("wait for page load")
            else:
                processed_tasks.append(task)
        
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
                        
                        # Check for task completion
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
        from together import Together
        if not config.TOGETHER_API_KEY:
            print("✗ Together.ai API key not configured")
            print("Update your API key in config.py")
            return False
        
        # Test Together.ai connection
        client = Together(
            api_key=config.TOGETHER_API_KEY,
            chat_model=config.DEFAULT_CHAT_MODEL,
            vision_model=config.DEFAULT_VISION_MODEL
        )
        models = client.get_models()
        if not models:
            print("✗ Together.ai API key invalid or no models available")
            print("Check your API key in config.py")
            return False
            
        print("✓ Together.ai connection successful")
        print(f"✓ Available models: {len(models)}")
        
        # Verify required models are available
        required_models = [config.DEFAULT_CHAT_MODEL, config.DEFAULT_VISION_MODEL]
        missing_models = []
        
        for required_model in required_models:
            if not any(required_model.lower() in model.get("name", "").lower() for model in models):
                missing_models.append(required_model)
        
        if missing_models:
            print(f"✗ Required models not available: {', '.join(missing_models)}")
            return False
            
        print("✓ Required models are available")
        
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
        print(f"✗ Together.ai setup failed: {str(e)}")
        print("Check your API key and internet connection")
        return False

# Main function to run the application
def main():
    """Main function with enhanced error checking and setup."""
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
        print("3. Configure your Together.ai API key in config.py")
        print("4. Check your internet connection")
        print("5. Check system resources (CPU, memory)")
        print("6. Try restarting your computer")

if __name__ == "__main__":
    main()
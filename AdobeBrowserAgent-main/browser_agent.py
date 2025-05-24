import base64
import json
import time
from io import BytesIO
from typing import Dict, Any, Optional, Tuple, List

# Import required libraries
from playwright.sync_api import sync_playwright, Page, Browser
from langchain_ollama.llms import OllamaLLM
from langchain.agents import Tool, AgentExecutor
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.runnable import RunnablePassthrough
from langchain.tools.render import format_tool_to_openai_function
import gradio as gr
from PIL import Image
from langchain_core.utils.function_calling import convert_to_openai_function

# Define the WebPerceptionAgent class
class WebPerceptionAgent:
    """An agent that perceives and interacts with web pages using a VLM."""
    
    def __init__(self, headless: bool = True, model_name: str = "llava:latest"):
        """
        Initialize the web perception agent.
        
        Args:
            headless: Whether to run the browser in headless mode
            model_name: The Ollama VLM model name to use (llava or bakllava)
        """
        self.headless = headless
        self.model_name = model_name
        self.browser = None
        self.page = None
        self.vlm = None
        self.agent_executor = None
        self.playwright = None
        
    def setup(self):
        """Set up the browser, VLM, and agent components."""
        try:
            # Check if Ollama is running and model is available
            import requests
            try:
                response = requests.get("http://localhost:11434/api/tags")
                if response.status_code != 200:
                    raise Exception("Ollama is not running")
                models = response.json().get("models", [])
                if not any(m["name"] == self.model_name for m in models):
                    # Try with :latest suffix
                    if not any(m["name"] == f"{self.model_name}:latest" for m in models):
                        raise Exception(f"Model {self.model_name} is not available in Ollama")
                    self.model_name = f"{self.model_name}:latest"
            except Exception as e:
                raise Exception(f"Ollama check failed: {str(e)}. Please ensure Ollama is running and {self.model_name} model is installed.")

            # Set up Playwright browser
            self.playwright = sync_playwright().start()
            self.browser = self.playwright.chromium.launch(headless=self.headless)
            self.page = self.browser.new_page()
            
            # Set up Ollama VLM
            self.vlm = OllamaLLM(model=self.model_name)
            
            # Create tools for the agent
            tools = [
                Tool(
                    name="GetPageState",
                    func=self.get_page_state,
                    description="Get the current webpage state (HTML and screenshot)"
                ),
                Tool(
                    name="ExecuteAction",
                    func=self.execute_action,
                    description="Execute an action on the webpage. Input should be a JSON string with 'action', 'selector', and 'text' keys."
                )
            ]
            
            # Create the agent executor
            self.create_agent_executor(tools)
            
            return self
        except Exception as e:
            self.close()  # Clean up resources if setup fails
            raise Exception(f"Setup failed: {str(e)}")
        
        
    def create_agent_executor(self, tools):
        """Create the LangChain agent executor with the given tools."""
        # Convert tools to OpenAI functions format using the new method
        functions = [convert_to_openai_function(t) for t in tools]
        
        # Create the prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a web agent with visual perception capabilities. "
                      "You can see web pages through screenshots and understand their HTML structure. "
                      "Your goal is to complete tasks by interacting with web pages."),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        # Create the agent
        llm_with_tools = self.vlm.bind(functions=functions)
        
        agent = (
            {
                "input": RunnablePassthrough(),
                "agent_scratchpad": lambda x: format_to_openai_function_messages(x["intermediate_steps"]),
            }
            | prompt
            | llm_with_tools
            | OpenAIFunctionsAgentOutputParser()
        )
        
        # Create the agent executor
        self.agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
        
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
            "screenshot_b64": screenshot_b64
        }
    
    def construct_multimodal_prompt(self, page_state: Dict[str, Any], task: str) -> Tuple[str, Any]:
        """
        Construct a multimodal prompt for the VLM.
        
        Args:
            page_state: The page state with HTML and screenshot
            task: The task description
            
        Returns:
            A prompt string for the VLM and the VLM with image bound
        """
        prompt = f"""
        System: You are an assistant with vision. Use the screenshot and HTML to decide the next action.
        
        User: <image>
        
        HTML: {page_state["html"][:10000]}  # Truncate HTML if too long
        
        Task: {task}
        
        Analyze the page and output one of:
        {{"action":"click"/"type"/"navigate", "selector":"CSS selector", "text":"optional text to type"}}.
        
        Assistant:
        """
        
        # Bind the image to the prompt
        llm_with_img = self.vlm.bind(images=[page_state["screenshot_b64"]])
        
        return prompt, llm_with_img
    
    def execute_action(self, action_json_str: str) -> str:
        """Execute an action on the webpage."""
        try:
            action_data = json.loads(action_json_str)
            
            action = action_data.get("action", "").lower()
            selector = action_data.get("selector", "")
            text = action_data.get("text", "")
            value = action_data.get("value", "")
            key = action_data.get("key", "")
            timeout = action_data.get("timeout", 30000)
            
            if action == "click":
                # Enhanced click handling with better selector strategies
                try:
                    print(f"\nClick action details:")
                    print(f"- Original selector: {selector}")
                    
                    # Track if this is a "first link" action
                    is_first_link = "first" in text.lower()
                    
                    # Enhance selector for better element targeting
                    if is_first_link:
                        # Prioritize navigation-focused selectors for first link
                        enhanced_selectors = [
                            "a[href]:visible:not([href^='#']):not([href^='javascript:']):first",  # First real link
                            ".search-result a[href]:visible:first",  # First search result
                            "h1 a[href]:visible:first, h2 a[href]:visible:first",  # First heading link
                            "article a[href]:visible:first",  # First article link
                            ".title a[href]:visible:first"  # First title link
                        ]
                        selector = ", ".join(enhanced_selectors)
                        print(f"- Enhanced selector for first navigable link: {selector}")
                    
                    # Wait for any element matching the selector to be visible
                    print("- Waiting for elements to be available...")
                    elements = self.page.locator(selector)
                    
                    # For first link, only get the first element
                    if is_first_link:
                        try:
                            first_element = elements.first
                            if first_element.is_visible():
                                element = first_element
                                try:
                                    href = element.get_attribute("href")
                                    text_content = element.text_content()
                                    print(f"- Found first link: {text_content[:100]}")
                                    print(f"- Target URL: {href}")
                                    
                                    # Verify this is a valid navigation URL
                                    if href and not href.startswith(('#', 'javascript:', 'mailto:', 'tel:')):
                                        # Try to navigate
                                        try:
                                            print("- Attempting navigation...")
                                            with self.page.expect_navigation(timeout=5000, wait_until="domcontentloaded") as nav:
                                                element.click(timeout=5000)
                                                nav.value  # Wait for navigation
                                            
                                            # Additional verification of navigation
                                            current_url = self.page.url
                                            print(f"- Navigation complete. New URL: {current_url}")
                                            
                                            # Wait for page to stabilize
                                            self.page.wait_for_load_state("domcontentloaded", timeout=5000)
                                            time.sleep(1)  # Short wait for dynamic content
                                            
                                            return f"Successfully navigated to first link: {current_url}"
                                        except Exception as nav_error:
                                            print(f"- Navigation failed: {str(nav_error)}")
                                            # Try direct navigation as fallback
                                            try:
                                                self.page.goto(href, wait_until="domcontentloaded", timeout=5000)
                                                return f"Successfully navigated to first link (fallback): {href}"
                                            except Exception as goto_error:
                                                return f"Error: Navigation failed - {str(goto_error)}"
                                    else:
                                        return f"Error: First link has invalid URL: {href}"
                                except Exception as e:
                                    return f"Error processing first link: {str(e)}"
                            else:
                                return "Error: First link is not visible"
                        except Exception as e:
                            return f"Error finding first link: {str(e)}"
                    
                    # For non-first link actions, continue with existing logic...
                    count = elements.count()
                    print(f"- Found {count} matching elements")
                    
                    if count == 0:
                        # Try alternative selectors if no elements found
                        alt_selectors = [
                            f"a:has-text('{text}')",  # Link containing text
                            f"text='{text}' >> a",  # Text within link
                            f"a:has-text('{text}') >> visible=true",  # Visible link with text
                            f"a[href*='{text.lower()}']"  # Link with text in URL
                        ]
                        
                        for alt_selector in alt_selectors:
                            print(f"- Trying alternative selector: {alt_selector}")
                            elements = self.page.locator(alt_selector)
                            count = elements.count()
                            if count > 0:
                                selector = alt_selector
                                print(f"- Found {count} elements with alternative selector")
                                break
                        
                        if count == 0:
                            return "Error: No clickable links found matching any selectors"
                    
                    # Try to find and click the most relevant visible element
                    success = False
                    error_messages = []
                    
                    for i in range(count):
                        element = elements.nth(i)
                        try:
                            if element.is_visible():
                                # Get element details for logging
                                try:
                                    element_text = element.text_content() or element.get_attribute("value") or ""
                                    href = element.get_attribute("href") or ""
                                    print(f"- Attempting to navigate to: {href}")
                                    print(f"- Link text: {element_text[:100]}")
                                except:
                                    print("- Attempting to click element (details unavailable)")
                                
                                # Multiple navigation strategies
                                try:
                                    # Strategy 1: Wait for navigation after click
                                    with self.page.expect_navigation(timeout=5000, wait_until="domcontentloaded") as navigation_info:
                                        element.click(timeout=5000, force=False)
                                        navigation_info.value  # Wait for navigation to complete
                                    success = True
                                    print("- Navigation successful")
                                    break
                                except Exception as e1:
                                    try:
                                        # Strategy 2: Force click with navigation wait
                                        with self.page.expect_navigation(timeout=5000, wait_until="domcontentloaded") as navigation_info:
                                            element.click(timeout=5000, force=True)
                                            navigation_info.value  # Wait for navigation to complete
                                        success = True
                                        print("- Navigation successful (force click)")
                                        break
                                    except Exception as e2:
                                        try:
                                            # Strategy 3: JavaScript navigation
                                            href = element.get_attribute("href")
                                            if href:
                                                print(f"- Attempting JavaScript navigation to: {href}")
                                                self.page.goto(href, wait_until="domcontentloaded", timeout=5000)
                                                success = True
                                                print("- Navigation successful (JavaScript)")
                                                break
                                        except Exception as e3:
                                            error_messages.append(f"Navigation attempts failed: {str(e1)} | {str(e2)} | {str(e3)}")
                                            continue
                        except Exception as e:
                            error_messages.append(f"Element {i} error: {str(e)}")
                            continue
                    
                    if success:
                        # Additional wait for page stability
                        try:
                            print("- Waiting for page to stabilize after navigation...")
                            self.page.wait_for_load_state("domcontentloaded", timeout=5000)
                            time.sleep(1)  # Short additional wait for dynamic content
                        except Exception as e:
                            print(f"- Note: Page stabilization timeout: {str(e)}")
                        return f"Successfully navigated to link from selector: {selector}"
                    else:
                        error_detail = "\n".join(error_messages)
                        return f"Error: Could not navigate to any matching links.\nDetails:\n{error_detail}"
                    
                except Exception as e:
                    print(f"- Navigation action failed: {str(e)}")
                    return f"Error executing navigation action: {str(e)}"
                
            elif action == "type":
                # Enhanced type handling
                try:
                    # Try to find the search input with more specific selectors
                    if not selector:
                        # Prioritized list of search input selectors
                        search_selectors = [
                            "input[type='search']",
                            "#searchInput",  # Wikipedia specific
                            "input[name='search']",
                            "input[placeholder*='search' i]",
                            "input[aria-label*='search' i]",
                            ".search-input",
                            "#search",
                            "input[type='text']"
                        ]
                        
                        for sel in search_selectors:
                            try:
                                element = self.page.locator(sel).first
                                if element.is_visible():
                                    selector = sel
                                    print(f"Found visible search input with selector: {sel}")
                                    break
                            except:
                                continue
                    
                    if not selector:
                        return "Error: Could not find suitable input field"
                    
                    print(f"Using selector for typing: {selector}")
                    # Wait for the element and ensure it's visible
                    element = self.page.locator(selector).first
                    element.wait_for(state="visible", timeout=5000)
                    
                    # Click the element first to ensure focus
                    try:
                        element.click(timeout=5000)
                        time.sleep(0.5)  # Short wait after click
                    except Exception as click_error:
                        print(f"Warning: Could not click input field: {str(click_error)}")
                    
                    # Clear existing text
                    element.fill("")
                    time.sleep(0.5)  # Wait after clearing
                    
                    # Type the text with delay between characters
                    element.type(text, delay=100)  # Slower typing for stability
                    time.sleep(0.5)  # Wait after typing
                    
                    # Press Enter if it's a search field
                    if any(s in selector.lower() for s in ["search", "query", "q"]):
                        print("Pressing Enter to submit search")
                        element.press("Enter")
                        time.sleep(1)  # Wait after submit
                        return f"Typed '{text}' into search and submitted"
                    return f"Typed '{text}' into input"
                    
                except Exception as e:
                    print(f"Type action error: {str(e)}")
                    return f"Error executing type action: {str(e)}"
                    
            elif action == "navigate":
                # Ensure we have a valid URL
                if not text:
                    text = "https://www.deeplearning.ai/courses"
                try:
                    self.page.goto(text, timeout=timeout)
                    return f"Navigated to URL: {text}"
                except Exception as e:
                    return f"Error navigating to URL: {str(e)}"
                
            elif action == "scroll":
                try:
                    if text.lower() == "top":
                        self.page.evaluate("window.scrollTo(0, 0)")
                        return "Scrolled to top of page"
                    elif text.lower() == "bottom":
                        self.page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                        return "Scrolled to bottom of page"
                    else:
                        # Try to scroll element into view
                        element = self.page.locator(selector).first
                        element.scroll_into_view_if_needed(timeout=timeout)
                        return f"Scrolled element into view: {selector}"
                except Exception as e:
                    return f"Error executing scroll action: {str(e)}"
                
            elif action == "wait":
                try:
                    if selector:
                        self.page.wait_for_selector(selector, timeout=timeout)
                        return f"Waited for element: {selector}"
                    else:
                        wait_time = float(text) if text else 1
                        time.sleep(wait_time)
                        return f"Waited for {wait_time} seconds"
                except Exception as e:
                    return f"Error executing wait action: {str(e)}"
            
            else:
                return f"Unknown action: {action}"
                
        except Exception as e:
            return f"Error executing action: {str(e)}"
    
    def analyze_page_and_decide(self, task: str) -> Tuple[Dict[str, Any], str]:
        """
        Analyze the current page and decide on the next action.
        
        Args:
            task: The task description
            
        Returns:
            The next action to execute and the VLM's reasoning
        """
        # Get page state
        page_state = self.get_page_state()
        
        # Construct a more specific prompt for task execution
        prompt = f"""
        System: You are a web agent with vision capabilities. Your responses must be in valid JSON format.
        DO NOT provide explanations or descriptions - ONLY output a single JSON object.

        Current State:
        - URL: {self.page.url}
        - Task: {task}
        - Screenshot and HTML are provided for analysis

        Required: Output ONLY ONE of these JSON formats (no other text):
        1. {{"action": "navigate", "text": "URL"}}
        2. {{"action": "click", "selector": "CSS_SELECTOR"}}
        3. {{"action": "type", "selector": "CSS_SELECTOR", "text": "TEXT_TO_TYPE"}}
        4. {{"action": "scroll", "text": "top"}} or {{"action": "scroll", "text": "bottom"}}
        5. {{"action": "wait", "selector": "CSS_SELECTOR"}} or {{"action": "wait", "text": "SECONDS"}}

        User: <image>
        HTML: {page_state["html"][:10000]}

        Assistant:"""
        
        # Bind the image to the prompt
        llm_with_img = self.vlm.bind(images=[page_state["screenshot_b64"]])
        
        try:
            # Get VLM response
            response = llm_with_img.invoke(prompt)
            
            # Try to extract JSON action
            import re
            import json
            
            # Look for JSON-like structure in the response
            # First try to find complete JSON objects
            json_pattern = r'\{(?:[^{}]|(?R))*\}'
            json_matches = re.findall(json_pattern, response, re.DOTALL)
            
            if not json_matches:
                # If no complete JSON found, try to extract key-value pairs
                action_match = re.search(r'"action"\s*:\s*"([^"]+)"', response)
                selector_match = re.search(r'"selector"\s*:\s*"([^"]+)"', response)
                text_match = re.search(r'"text"\s*:\s*"([^"]+)"', response)
                
                if action_match:
                    action_data = {
                        "action": action_match.group(1)
                    }
                    if selector_match:
                        action_data["selector"] = selector_match.group(1)
                    if text_match:
                        action_data["text"] = text_match.group(1)
                        
                    # Validate the constructed action
                    if self._validate_action(action_data):
                        return action_data, response
            
            # Try parsing complete JSON objects
            for json_str in json_matches:
                try:
                    action_data = json.loads(json_str)
                    if self._validate_action(action_data):
                        return action_data, response
                except json.JSONDecodeError:
                    continue
            
            # If no valid JSON found, use task-based fallback
            print("[DEBUG] No valid JSON action found in response:", response)
            return self._get_fallback_action(task)
            
        except Exception as e:
            print("[DEBUG] Error in analyze_page_and_decide:", str(e))
            return self._get_fallback_action(task)
            
    def _validate_action(self, action_data: Dict[str, Any]) -> bool:
        """Validate if the action data has all required fields."""
        if "action" not in action_data:
            return False
            
        action = action_data["action"].lower()
        
        # Validate action type and required fields
        if action == "navigate" and "text" in action_data:
            return True
        elif action == "click" and "selector" in action_data:
            return True
        elif action == "type" and "selector" in action_data and "text" in action_data:
            return True
        elif action == "scroll" and "text" in action_data:
            return True
        elif action == "wait" and ("selector" in action_data or "text" in action_data):
            return True
            
        return False
    
    def _get_fallback_action(self, task: str) -> Tuple[Dict[str, Any], str]:
        """Get a fallback action based on the task description."""
        tl = task.lower()
        
        # Extract potential search terms for type actions
        search_terms = None
        if "type" in tl:
            # Extract text between "type" and "and" or end of string
            type_idx = tl.index("type") + 4
            and_idx = tl.find(" and ", type_idx)
            if and_idx != -1:
                search_terms = tl[type_idx:and_idx].strip()
            else:
                search_terms = tl[type_idx:].strip()
        
        # Extract text to click (anything after "click" or "click on")
        click_text = None
        if "click" in tl:
            click_idx = tl.index("click") + 5
            if tl[click_idx:].startswith(" on "):
                click_idx += 4
            click_text = tl[click_idx:].strip()
            # Remove common endings
            for ending in [" and ", " then ", " next ", "."]:
                if click_text.endswith(ending):
                    click_text = click_text[:-len(ending)].strip()
        
        # Prioritized fallback actions
        if task.startswith(("http://", "https://")):
            return {"action": "navigate", "text": task}, "fallback: direct URL navigation"
        elif search_terms:
            # Handle explicit type commands first
            return {
                "action": "type",
                "selector": "#searchInput",  # Try Wikipedia's search input first
                "text": search_terms
            }, f"fallback: type action with text: {search_terms}"
        elif "click" in tl and "first" in tl:
            # Enhanced first-click selector
            return {
                "action": "click",
                "selector": "a:visible, button:visible, [role='button']:visible, .search-result a:visible",
                "text": "first"  # Indicate this is a first-element click
            }, "fallback: click first visible interactive element"
        elif click_text:
            # Click with extracted text
            return {
                "action": "click",
                "selector": f"text='{click_text}'",
                "text": click_text
            }, f"fallback: click element with text: {click_text}"
        elif "rag" in tl.lower():
            return {
                "action": "click",
                "selector": "a:has-text('RAG'), a:has-text('Retrieval'), a:has-text('Augmented'), .course-card:has-text('RAG')",
                "text": "RAG"
            }, "fallback: click RAG-related link"
        elif "search" in tl:
            return {
                "action": "type",
                "selector": "input[type='search']",
                "text": task
            }, "fallback: search action"
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
        # Split on common task separators
        separators = [" then ", " and then ", " after that ", ", then ", "; ", " next ", " followed by "]
        tasks = [task_description]
        
        for separator in separators:
            new_tasks = []
            for task in tasks:
                new_tasks.extend(t.strip() for t in task.split(separator))
            tasks = new_tasks
            
        # Filter out empty tasks and normalize
        tasks = [t.strip() for t in tasks if t.strip()]
        print(f"Parsed tasks: {tasks}")
        return tasks

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
                        
                        # Execute the action
                        print(f"Executing action: {action['action']}")
                        action_json_str = json.dumps(action)
                        result = self.execute_action(action_json_str)
                        results.append(f"Subtask {subtask_index} - Step {step}: {result}")
                        print(f"Action result: {result}")
                        
                        # Wait for page to load/react with reduced timeout
                        try:
                            print("Waiting for page to stabilize...")
                            self.page.wait_for_load_state("domcontentloaded", timeout=5000)
                            print("✓ Page stable")
                        except Exception as e:
                            print(f"Warning: Page load wait timed out: {str(e)}")
                        
                        # Wait for any animations to complete
                        time.sleep(1)
                        
                        # Check for error states
                        if "error" in result.lower():
                            print(f"Error detected in action result: {result}")
                            results.append(f"Subtask {subtask_index} - Step {step}: Action failed. Trying next step.")
                            continue
                        
                    except Exception as step_error:
                        print(f"Error in subtask {subtask_index} step {step}: {str(step_error)}")
                        results.append(f"Subtask {subtask_index} - Step {step}: Error: {str(step_error)}")
                        continue
                
            return "\n".join(results)
            
        except Exception as e:
            return f"Task execution failed: {str(e)}"
    
    def close(self):
        """Close the browser and clean up resources."""
        if self.browser:
            self.browser.close()
        if self.playwright:
            self.playwright.stop()
            
    def __enter__(self):
        return self.setup()
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

# Define the Gradio interface class
class WebAgentGradioInterface:
    """A Gradio interface for the web agent with visual perception."""
    
    def __init__(self):
        """Initialize the web agent Gradio interface."""
        self.agent = None
        self.is_running = False
        self.debug_mode = True  # Enable debug mode by default
    
    def setup_agent(self, headless=True, model_name="llava:latest"):
        """Set up the web agent with enhanced error checking."""
        if self.agent is None:
            try:
                print("\n=== Setting up Web Agent ===")
                
                # Check Ollama status
                print("Checking Ollama status...")
                import requests
                try:
                    response = requests.get("http://localhost:11434/api/tags")
                    if response.status_code != 200:
                        raise Exception("Ollama is not running properly")
                    print("✓ Ollama is running")
                    
                    # Verify model availability
                    models = response.json().get("models", [])
                    if not any(m["name"].startswith(model_name.split(":")[0]) for m in models):
                        raise Exception(f"Model {model_name} is not available")
                    print(f"✓ Model {model_name} is available")
                    
                except requests.exceptions.ConnectionError:
                    raise Exception("Could not connect to Ollama. Is it running?")
                
                # Initialize the agent
                print("Initializing agent...")
                self.agent = WebPerceptionAgent(headless=headless, model_name=model_name)
                
                # Setup the agent
                print("Setting up agent components...")
                self.agent.setup()
                print("✓ Agent setup complete")
                
                return self.agent
                
            except Exception as e:
                detailed_error = f"Failed to set up web agent: {str(e)}\n"
                detailed_error += "Please check:\n"
                detailed_error += "1. Is Ollama running? (ollama serve)\n"
                detailed_error += "2. Is the LLaVA model installed? (ollama pull llava)\n"
                detailed_error += "3. Are all dependencies installed? (pip install -r requirements.txt)\n"
                detailed_error += "4. Are Playwright browsers installed? (playwright install)"
                print(detailed_error)
                raise gr.Error(detailed_error)
    
    def run_agent(self, url, task, max_steps=10, show_progress=True):
        """Run the web agent with enhanced error handling and diagnostics."""
        if not url or not task:
            raise gr.Error("Please provide both URL and task description")
            
        if not url.startswith(("http://", "https://")):
            url = "https://" + url
            
        if self.is_running:
            return "Agent is already running. Please wait.", []
            
        self.is_running = True
        result_text = ""
        screenshots = []
        
        try:
            print("\n=== Starting New Task ===")
            print(f"URL: {url}")
            print(f"Task: {task}")
            print(f"Max steps: {max_steps}")
            
            # Set up agent if needed
            agent = self.setup_agent(headless=False)
            
            # Run the task with progress updates
            print("\n=== Executing Task ===")
            result_text = agent.run_task(url, task, max_steps)
            
            if show_progress:
                try:
                    print("\n=== Capturing Final State ===")
                    screenshot = self.capture_screenshot(agent)
                    if screenshot:
                        screenshots.append(screenshot)
                        print("✓ Screenshot captured")
                    else:
                        print("Warning: Could not capture screenshot")
                except Exception as e:
                    print(f"Warning: Screenshot capture failed: {str(e)}")
            
            return result_text, screenshots
            
        except Exception as e:
            error_msg = f"Error running web agent: {str(e)}\n"
            error_msg += "\nTroubleshooting steps:\n"
            error_msg += "1. Check your internet connection\n"
            error_msg += "2. Verify the URL is accessible\n"
            error_msg += "3. Try refreshing the page\n"
            error_msg += "4. Check if the website is responding"
            print(error_msg)
            return error_msg, screenshots
        finally:
            self.is_running = False
            print("\n=== Task Execution Complete ===")
    
    def capture_screenshot(self, agent):
        """Capture a screenshot from the agent's page and convert to PIL Image."""
        try:
            screenshot_buffer = agent.page.screenshot()
            return Image.open(BytesIO(screenshot_buffer))
        except Exception as e:
            print(f"Warning: Screenshot capture failed: {str(e)}")
            return None

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
                        placeholder="Search for 'machine learning' and click the first result",
                        info="Describe the task you want the agent to perform",
                        lines=3
                    )
                    
                    max_steps = gr.Slider(
                        minimum=1,
                        maximum=20,
                        value=10,
                        step=1,
                        label="Maximum Steps",
                        info="Maximum number of actions the agent will take"
                    )
                    
                    run_button = gr.Button("Run Web Agent", variant="primary")
                    
                with gr.Column(scale=3):
                    result_text = gr.Textbox(
                        label="Results",
                        lines=20,
                        info="Progress and results of the web agent"
                    )
                    
                    result_gallery = gr.Gallery(
                        label="Screenshots",
                        columns=2,
                        object_fit="contain",
                        height="500px"
                    )
            
            # Set up the run button to call the agent
            run_button.click(
                fn=self.run_agent,
                inputs=[url_input, task_input, max_steps],
                outputs=[result_text, result_gallery]
            )
            
        return interface
    
    def launch(self, **kwargs):
        """Create and launch the Gradio interface."""
        interface = self.create_interface()
        interface.queue()  # Enable queuing for the interface
        
        # Set default kwargs if not provided
        if "share" not in kwargs:
            kwargs["share"] = False
        if "inbrowser" not in kwargs:
            kwargs["inbrowser"] = True
            
        try:
            interface.launch(**kwargs)
        finally:
            self.cleanup_agent()

def check_system_requirements():
    """Check if all system requirements are met."""
    try:
        print("\n=== System Requirements Check ===")
        
        # Check Python version
        import sys
        print(f"Python version: {sys.version}")
        
        # Check Playwright
        try:
            from playwright.sync_api import sync_playwright
            print("✓ Playwright is installed")
            
            # Test browser launch
            with sync_playwright() as p:
                try:
                    browser = p.chromium.launch(headless=True)
                    browser.close()
                    print("✓ Browser launch test successful")
                except Exception as e:
                    print(f"✗ Browser launch failed: {str(e)}")
                    print("Try running: playwright install")
                    return False
        except ImportError:
            print("✗ Playwright not installed")
            print("Try running: pip install playwright")
            return False
        
        # Check Ollama
        import requests
        try:
            response = requests.get("http://localhost:11434/api/tags")
            if response.status_code == 200:
                print("✓ Ollama is running")
                # Check LLaVA model
                models = response.json().get("models", [])
                model_names = [m["name"].split(":")[0] for m in models]
                if "llava" in model_names:
                    print("✓ LLaVA model is available")
                else:
                    print("✗ LLaVA model not found")
                    print("Try running: ollama pull llava")
                    return False
            else:
                print("✗ Ollama is not responding correctly")
                print("Check if Ollama is running with: ollama serve")
                return False
        except requests.exceptions.ConnectionError:
            print("✗ Could not connect to Ollama")
            print("Start Ollama with: ollama serve")
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
        print(f"\n✗ Error during system check: {str(e)}")
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
        print("3. Start Ollama: ollama serve")
        print("4. Pull LLaVA model: ollama pull llava")
        print("5. Check system resources (CPU, memory)")
        print("6. Try restarting your computer")

if __name__ == "__main__":
    main()
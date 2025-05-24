import base64
import json
import time
from io import BytesIO
from typing import Dict, Any, Optional, Tuple

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
    
    def __init__(self, headless: bool = True, model_name: str = "meta-llama/Llama-Vision-Free"):
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
        """
        Execute an action on the webpage.
        
        Args:
            action_json_str: A JSON string with action details containing:
                - action: One of ["click", "type", "navigate", "select", "hover", "scroll", "wait", "press"]
                - selector: CSS selector for the element (except for navigate)
                - text: Text to type or URL to navigate to
                - value: Value for select elements
                - key: Key to press (for press action)
                - timeout: Optional timeout in milliseconds
        
        Returns:
            A message indicating the result of the action
        """
        try:
            action_data = json.loads(action_json_str)
            
            action = action_data.get("action", "").lower()
            selector = action_data.get("selector", "")
            text = action_data.get("text", "")
            value = action_data.get("value", "")
            key = action_data.get("key", "")
            timeout = action_data.get("timeout", 30000)  # Default 30 seconds timeout
            
            if action == "click":
                # Click an element
                self.page.click(selector, timeout=timeout)
                return f"Clicked element with selector: {selector}"
                
            elif action == "type":
                # Try to find the search input
                if not selector:
                    # Try different possible search input selectors
                    search_selectors = [
                        "input[type='search']",
                        "input[placeholder*='search' i]",
                        "input[aria-label*='search' i]",
                        "input.search",
                        "#search",
                        ".search-input"
                    ]
                    
                    for sel in search_selectors:
                        try:
                            if self.page.locator(sel).count() > 0:
                                selector = sel
                                break
                        except:
                            continue
                
                if selector:
                    self.page.fill(selector, text)
                    # Press Enter to submit search
                    self.page.press(selector, "Enter")
                    return f"Typed '{text}' into search and submitted"
                else:
                    return "Error: Could not find search input"
                
            elif action == "navigate":
                # Ensure we have a valid URL
                if not text:
                    text = "https://www.deeplearning.ai/courses"
                self.page.goto(text)
                return f"Navigated to URL: {text}"
                
            elif action == "select":
                # Select an option from a dropdown
                self.page.select_option(selector, value=value, timeout=timeout)
                return f"Selected option '{value}' in element with selector: {selector}"
                
            elif action == "hover":
                # Hover over an element
                self.page.hover(selector, timeout=timeout)
                return f"Hovered over element with selector: {selector}"
                
            elif action == "scroll":
                # full-page scroll
                direction = text.lower()
                if direction == "bottom":
                    self.page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                    return "Scrolled to bottom"
                elif direction == "top":
                    self.page.evaluate("window.scrollTo(0, 0)")
                    return "Scrolled to top"
                else:
                    # if they ever supply a CSS selector, still scroll that element into view:
                    self.page.locator(selector).scroll_into_view_if_needed(timeout=timeout)
                    return f"Scrolled to element {selector}"
                

            elif action == "wait":
                # Wait for a specific condition
                if selector:
                    # Wait for element
                    self.page.wait_for_selector(selector, timeout=timeout)
                    return f"Waited for element with selector: {selector}"
                else:
                    # Wait for specified time
                    time.sleep(float(text) if text else 1)
                    return f"Waited for {text or 1} seconds"
                
            elif action == "press":
                # Press a key
                self.page.press(selector, key, timeout=timeout)
                return f"Pressed key '{key}' on element with selector: {selector}"
                
            elif action == "check":
                # Check a checkbox
                self.page.check(selector, timeout=timeout)
                return f"Checked element with selector: {selector}"
                
            elif action == "uncheck":
                # Uncheck a checkbox
                self.page.uncheck(selector, timeout=timeout)
                return f"Unchecked element with selector: {selector}"
                
            elif action == "clear":
                # Clear input field
                self.page.fill(selector, "", timeout=timeout)
                return f"Cleared element with selector: {selector}"
                
            elif action == "double_click":
                # Double click an element
                self.page.dblclick(selector, timeout=timeout)
                return f"Double clicked element with selector: {selector}"
                
            elif action == "right_click":
                # Right click an element
                self.page.click(selector, button="right", timeout=timeout)
                return f"Right clicked element with selector: {selector}"
                
            else:
                return f"Unknown action: {action}. Supported actions are: click, type, navigate, select, hover, scroll, wait, press, check, uncheck, clear, double_click, right_click"
                
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
        
        # Construct a more specific prompt for course search
        prompt = f"""
            You are a vision-based browser automation agent.

            Task: {task}

            Use the screenshot and HTML to choose the next action.

            **Respond with ONLY one flat JSON object** in this exact format:
            {{
            "action": "click" | "type" | "navigate" | "scroll",
            "selector": "CSS selector (omit for navigate/scroll)",
            "text": "text to type, URL to visit, or 'bottom'/'top' for scroll"
            }}

            ❌ No markdown/code fences  
            ❌ No extra keys (parameters, arguments, intent…)  
            ✅ Just the JSON object.

            Screenshot: <image>  
            HTML: {page_state["html"][:10000]}  
            Current URL: {self.page.url}
            """



        # Bind the image to the prompt
        llm_with_img = self.vlm.bind(images=[page_state["screenshot_b64"]])
        
        # Get VLM response
        response = llm_with_img.invoke(prompt)
        import re
        response = re.sub(r"```(?:json|python)?", "", response, flags=re.IGNORECASE).strip("`").strip()
        print("[DEBUG] Raw VLM response:", response)
        
        # Try to extract JSON action
        try:
    # find the first {...}
            match = re.search(r"(\{[\s\S]*?\}|\[[\s\S]*?\])", response)
            if not match:
                raise ValueError("No JSON object found")

            action = json.loads(match.group(0))

            # flatten nested blocks if they appear
            for nested in ("parameters", "arguments"):
                if nested in action and isinstance(action[nested], dict):
                    action.update(action.pop(nested))
            
            synonyms = {
                "movetourl": "navigate",
                "movetourl": "navigate",
                "gotourl":  "navigate",
                "open":     "navigate",
            }


            act = action.get("action", "").lower()
            if act in synonyms:
                action["action"] = synonyms[act]
                # grab URL if provided under a different key
                action["text"] = action.get("url", action.get("text", self.page.url))
                # remove extraneous keys
                action.pop("url", None)

            # Refresh act after any remapping
            act = action.get("action","").lower()
            if act not in {"click","type","navigate","scroll","select","hover","wait","press",
                        "check","uncheck","clear","double_click","right_click"}:
                raise ValueError(f"Unsupported action: {act}")

            # fill in missing defaults
            if act == "type" and not action.get("text"):
                action["text"] = task
            if act == "navigate" and not action.get("text"):
                action["text"] = self.page.url
            if act == "scroll" and not action.get("text"):
                action["text"] = "bottom"

            return action, response

        except Exception as e:
            print("[DEBUG] parse error:", e)
            # task-based fallback
            tl = task.lower()
            if "search" in tl:
                return {"action":"type","selector":"input[type='search']","text":task}, f"fallback: {e}"
            if "submit" in tl:
                return {"action":"click","selector":"input[type='submit']"}, f"fallback: {e}"
            if "scroll" in tl:
                return {"action":"scroll","text":"bottom"}, f"fallback: {e}"
            if "navigate" in tl or task.startswith("http"):
                return {"action":"navigate","text":task}, f"fallback: {e}"
            if "click" in tl:
                return {"action": "click","selector": "a[href*='item?id=']"}, f"fallback : {e}"
            return {"action":"type","selector":"input","text":task}, f"fallback: {e}"


    def run_task(self, initial_url: str, task: str, max_steps: int = 10) -> str:
        """
        Run a task starting from the given URL.
        
        Args:
            initial_url: The URL to start from
            task: The task to perform
            max_steps: Maximum number of steps to take
            
        Returns:
            Final status report
        """
        # Navigate to initial URL
        self.page.goto(initial_url)
        print(f"Navigated to {initial_url}")
        
        # Loop until task completion or max steps reached
        step = 0
        results = []
        last_action = None
        consecutive_same_actions = 0
        
        while step < max_steps:
            step += 1
            print(f"\nStep {step}/{max_steps}")
            
            # Get current page state
            page_state = self.get_page_state()
            
            # Analyze page and decide next action
            action, reasoning = self.analyze_page_and_decide(task)
            
            # If no valid action, stop
            if action is None:
                results.append(f"Step {step}: Could not determine next action. VLM reasoning: {reasoning}")
                break
            
            # Check for task completion
            if "task complete" in reasoning.lower():
                results.append(f"Step {step}: Task completed successfully!")
                break
            
            # Check for stuck state (same action repeated too many times)
            if last_action == action:
                consecutive_same_actions += 1
                if consecutive_same_actions >= 3:
                    results.append(f"Step {step}: Detected potential stuck state. Same action repeated {consecutive_same_actions} times.")
                    break
            else:
                consecutive_same_actions = 0
                last_action = action
            
            # Execute the action
            action_json_str = json.dumps(action)
            result = self.execute_action(action_json_str)
            results.append(f"Step {step}: {result}")
            if (
            action.get("action") == "type"
            and task.lower() in action.get("text", "").lower()
            ):
                results.append(f"Step {step}: Search submitted; ending early.\n")
                break

        # ✅ Optional: stop if URL or content indicates search success
            if "search" in self.page.url.lower() or task.lower() in self.page.content().lower():
                results.append(f"Step {step}: Task keyword detected on page. Assuming task complete.\n")
                break
            
            # Wait for page to load/react
            try:
                # Wait for network to be idle
                self.page.wait_for_load_state("networkidle", timeout=5000)
            except Exception:
                # If timeout, continue anyway
                pass
            
            # Wait for any animations to complete
            time.sleep(1)
            
            # Check for error states
            if "error" in result.lower():
                results.append(f"Step {step}: Action failed. Stopping execution.")
                break
            
            # Check for navigation completion
            if action.get("action") == "navigate":
                try:
                    self.page.wait_for_load_state("domcontentloaded", timeout=10000)
                except Exception as e:
                    results.append(f"Step {step}: Navigation timeout or error: {str(e)}")
                    break
            
            # Take screenshot after action
            try:
                screenshot = self.page.screenshot()
                # You could save or process the screenshot here if needed
            except Exception as e:
                print(f"Warning: Could not take screenshot: {str(e)}")
            
            # Check for success indicators in the page
            try:
                # Look for common success indicators
                success_indicators = [
                    "success",
                    "completed",
                    "thank you",
                    "welcome",
                    "logged in",
                    "submitted"
                ]
                
                page_text = self.page.content().lower()
                if any(indicator in page_text for indicator in success_indicators):
                    results.append(f"Step {step}: Detected success indicators on page.")
                    break
                
            except Exception as e:
                print(f"Warning: Error checking success indicators: {str(e)}")
        
        # Return results
        return "\n".join(results)
    
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
    
    def setup_agent(self, headless=True, model_name="llava"):
        """Set up the web agent if not already initialized."""
        if self.agent is None:
            self.agent = WebPerceptionAgent(headless=headless, model_name=model_name).setup()
        return self.agent
    
    def cleanup_agent(self):
        """Clean up the agent resources."""
        if self.agent:
            self.agent.close()
            self.agent = None
    
    def run_agent(self, url, task, max_steps=10, show_progress=True):
        """
        Run the web agent with the given URL and task.
        
        Args:
            url: The starting URL
            task: The task description
            max_steps: Maximum number of steps to take
            show_progress: Whether to show progress updates
            
        Returns:
            A tuple of (result text, list of screenshot images)
        """
        # Check if we're already running
        if self.is_running:
            return "Agent is already running. Please wait.", []
        
        self.is_running = True
        result_text = ""
        screenshots = []
        
        try:
            # Set up agent if needed
            agent = self.setup_agent(headless=False)
            
            # Navigate to initial URL
            agent.page.goto(url)
            result_text += f"Navigated to {url}\n"
            
            if show_progress:
                # Capture initial screenshot
                initial_screenshot = self.capture_screenshot(agent)
                screenshots.append(initial_screenshot)
                
                # Yield initial update
                yield result_text, screenshots
            
            # Loop until task completion or max steps reached
            step = 0
            
            while step < max_steps:
                step += 1
                result_text += f"\nStep {step}/{max_steps}\n"
                
                if show_progress:
                    yield result_text, screenshots
                
                # Analyze page and decide action
                action, reasoning = agent.analyze_page_and_decide(task)
                
                # Show reasoning
                result_text += f"Analysis: {reasoning[:300]}...\n"
                
                # If no valid action, stop
                if action is None:
                    result_text += f"Could not determine next action.\n"
                    break
                    
                # Execute action
                action_json_str = json.dumps(action)
                execution_result = agent.execute_action(action_json_str)
                result_text += f"Action: {execution_result}\n"
                
                # Wait for page to load/react
                time.sleep(1)
                
                # Capture screenshot after action
                if show_progress:
                    screenshot = self.capture_screenshot(agent)
                    screenshots.append(screenshot)
                    yield result_text, screenshots
                
                # Check if task is complete
                if "task complete" in reasoning.lower():
                    result_text += "Task completed successfully!\n"
                    break
            
            # Final update
            result_text += "\nTask execution completed.\n"
            if show_progress:
                final_screenshot = self.capture_screenshot(agent)
                if final_screenshot not in screenshots:
                    screenshots.append(final_screenshot)
                
            return result_text, screenshots
            
        except Exception as e:
            result_text += f"\nError: {str(e)}\n"
            return result_text, screenshots
        finally:
            self.is_running = False
    
    def capture_screenshot(self, agent):
        """Capture a screenshot from the agent's page and convert to PIL Image."""
        screenshot_buffer = agent.page.screenshot()
        img = Image.open(BytesIO(screenshot_buffer))
        return img

    def create_interface(self):
        """Create and return the Gradio interface."""
        with gr.Blocks(title="Web Agent with Visual Perception") as interface:
            gr.Markdown("# Web Agent with Visual Perception")
            gr.Markdown("Enter a URL and task description to automate web interactions using visual AI.")
            
            with gr.Row():
                with gr.Column(scale=2):
                    url_input = gr.Textbox(
                        label="URL",
                        placeholder="https://example.com",
                        info="Enter the starting URL for the web agent"
                    )
                    
                    task_input = gr.Textbox(
                        label="Task Description",
                        placeholder="Log in with username 'demo' and password 'password123'",
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

# Main function to run the application
def main():
    """Main function to launch the web agent Gradio interface."""
    print("Starting Web Agent with Visual Perception...")
    print("This application requires Ollama with the llava model installed.")
    print("Launching Gradio interface...")
    
    app = WebAgentGradioInterface()
    app.launch()

if __name__ == "__main__":
    main()

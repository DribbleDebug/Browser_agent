"""
Together.ai API client implementation.

This client provides an interface to Together.ai's API services including:
- Chat completions (text generation)
- Image analysis (vision tasks)
- Model listing
- Streaming responses

Example Usage:
    # Initialize the client
    client = TogetherAIClient(api_key="your_api_key")
    
    # Simple chat
    response = client.chat(
        messages=[
            {"role": "user", "content": "What is the capital of France?"}
        ]
    )
    print(response)  # Paris is the capital of France...
    
    # Image analysis
    analysis = client.analyze_image(
        image_url="https://example.com/image.jpg",
        prompt="Describe what you see in this image"
    )
    print(analysis)  # The image shows...
"""

import requests
import json
from typing import Optional, Dict, Any, List, Generator

class TogetherAIClient:
    """Client for interacting with Together.ai API.
    
    This class provides methods to:
    1. Generate text using chat models
    2. Analyze images using vision models
    3. List available models
    4. Stream chat completions
    
    Attributes:
        api_key (str): Your Together.ai API key
        base_url (str): API base URL
        chat_model (str): Default chat model to use
        vision_model (str): Default vision model to use
        headers (dict): HTTP headers for API requests
    """
    
    def __init__(
        self,
        api_key: str,
        chat_model: str = None,
        vision_model: str = None,
        base_url: str = "https://api.together.xyz/v1"
    ):
        """Initialize the Together.ai client.
        
        Args:
            api_key: Your Together.ai API key
            chat_model: Optional default chat model (e.g., "meta-llama/Llama-2-70b-chat")
            vision_model: Optional default vision model
            base_url: Base URL for the API
            
        Example:
            client = TogetherAIClient(
                api_key="your_api_key",
                chat_model="meta-llama/Llama-2-70b-chat"
            )
        """
        self.api_key = api_key
        self.base_url = base_url
        self.chat_model = chat_model
        self.vision_model = vision_model
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    def get_models(self) -> list:
        """Get list of available models from Together.ai.
        
        Returns:
            list: Available models and their details
            
        Example:
            models = client.get_models()
            for model in models:
                print(f"Model: {model['name']}, Type: {model['type']}")
        """
        url = f"{self.base_url}/models"
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        return response.json()["models"]
    
    def analyze_image(
        self,
        image_url: str,
        prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024
    ) -> str:
        """Analyze an image using Together.ai vision model.
        
        Args:
            image_url: URL of the image to analyze
            prompt: Question or instruction about the image
            model: Vision model to use (optional)
            temperature: Creativity of response (0-1)
            max_tokens: Maximum length of response
            
        Returns:
            str: Model's analysis of the image
            
        Example:
            analysis = client.analyze_image(
                image_url="https://example.com/cat.jpg",
                prompt="What breed is this cat?",
                temperature=0.5
            )
            print(analysis)  # This appears to be a Siamese cat...
        """
        url = f"{self.base_url}/inference"
        model = model or self.vision_model
        
        # Prepare the API payload
        payload = {
            "model": model,
            "prompt": f"<image>{image_url}</image>\n{prompt}",
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False
        }
        
        # Make API request
        response = requests.post(url, headers=self.headers, json=payload)
        response.raise_for_status()
        return response.json()["output"]["choices"][0]["text"]
    
    def chat(
        self,
        messages: list,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024
    ) -> str:
        """Send a simple chat request to Together.ai.
        
        Args:
            messages: List of conversation messages
            model: Chat model to use (optional)
            temperature: Response creativity (0-1)
            max_tokens: Maximum response length
            
        Returns:
            str: Model's response text
            
        Example:
            response = client.chat([
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is Python?"}
            ])
            print(response)  # Python is a programming language...
        """
        url = f"{self.base_url}/chat/completions"
        model = model or self.chat_model
        
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False
        }
        
        response = requests.post(url, headers=self.headers, json=payload)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str = "meta-llama/Llama-2-70b-chat",
        temperature: float = 0.7,
        max_tokens: int = 1024,
        stop: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Send a chat completion request with full response details.
        
        This method returns the complete API response including metadata,
        while the chat() method returns just the response text.
        
        Args:
            messages: List of conversation messages
            model: Chat model to use
            temperature: Response creativity (0-1)
            max_tokens: Maximum response length
            stop: Optional list of stop sequences
            
        Returns:
            dict: Complete API response
            
        Example:
            response = client.chat_completion([
                {
                    "role": "system",
                    "content": "You are a web automation assistant."
                },
                {
                    "role": "user",
                    "content": "How do I click the login button?"
                }
            ])
            
            # Access different parts of the response
            text = response["choices"][0]["message"]["content"]
            model_used = response["model"]
            tokens_used = response["usage"]["total_tokens"]
        """
        endpoint = f"{self.base_url}/chat/completions"
        
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False
        }
        
        if stop:
            payload["stop"] = stop
        
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
    
    def stream_chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str = "meta-llama/Llama-2-70b-chat",
        temperature: float = 0.7,
        max_tokens: int = 1024
    ) -> Generator[str, None, None]:
        """Stream a chat completion response in real-time.
        
        This method yields response chunks as they arrive, useful for
        displaying responses in real-time or handling long generations.
        
        Args:
            messages: List of conversation messages
            model: Chat model to use
            temperature: Response creativity (0-1)
            max_tokens: Maximum response length
            
        Yields:
            str: Text chunks as they arrive
            
        Example:
            # Stream response in real-time
            for chunk in client.stream_chat_completion([
                {"role": "user", "content": "Write a long story"}
            ]):
                print(chunk, end="", flush=True)  # Print as text arrives
        """
        endpoint = f"{self.base_url}/chat/completions"
        
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True
        }
        
        try:
            # Make streaming request
            response = requests.post(
                endpoint,
                headers=self.headers,
                json=payload,
                stream=True
            )
            response.raise_for_status()
            
            # Process streaming response
            for line in response.iter_lines():
                if line:
                    try:
                        # Parse server-sent events
                        chunk = json.loads(line.decode("utf-8").replace("data: ", ""))
                        if chunk["choices"][0]["finish_reason"] is None:
                            yield chunk["choices"][0]["delta"]["content"]
                    except:
                        continue
                        
        except requests.exceptions.RequestException as e:
            raise Exception(f"Streaming chat completion failed: {str(e)}")

# Example usage and testing
if __name__ == "__main__":
    # This code runs when the file is run directly
    import os
    
    # Get API key from environment variable
    api_key = os.getenv("TOGETHER_API_KEY")
    if not api_key:
        print("Please set TOGETHER_API_KEY environment variable")
        exit(1)
    
    # Create client
    client = TogetherAIClient(api_key=api_key)
    
    # Test model listing
    print("\nAvailable models:")
    models = client.get_models()
    for model in models[:3]:  # Show first 3 models
        print(f"- {model['name']}")
    
    # Test chat
    print("\nTesting chat:")
    response = client.chat([
        {"role": "user", "content": "What is Together.ai?"}
    ])
    print(f"Response: {response[:100]}...")  # Show first 100 chars
    
    # Test streaming
    print("\nTesting streaming (first 3 chunks):")
    chunks = []
    for chunk in client.stream_chat_completion([
        {"role": "user", "content": "Count from 1 to 10"}
    ]):
        chunks.append(chunk)
        if len(chunks) >= 3:
            break
    print("Chunks received:", chunks) 
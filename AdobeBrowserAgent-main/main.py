"""Main entry point for the Web Perception Agent."""

from agent import WebAgentGradioInterface

def main():
    """Initialize and run the web agent interface."""
    try:
        print("\n=== Starting Web Agent Application ===")
        
        # Create and launch the interface
        app = WebAgentGradioInterface()
        app.launch(share=False, inbrowser=True)
        
    except Exception as e:
        print(f"\nError starting the application: {str(e)}")
        print("\nTroubleshooting steps:")
        print("1. Make sure all requirements are installed: pip install -r requirements.txt")
        print("2. Install Playwright browsers: playwright install")
        print("3. Check your Together.ai API key in config.py")
        print("4. Check your internet connection")
        print("5. Try restarting your computer")

if __name__ == "__main__":
    main() 
"""
Main entry point for the Web Agent with Visual Perception.
"""

from agent import WebAgentGradioInterface

def main():
    """Run the web agent interface."""
    interface = WebAgentGradioInterface()
    interface.launch(share=False)

if __name__ == "__main__":
    main() 
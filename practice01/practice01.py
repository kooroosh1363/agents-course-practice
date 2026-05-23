# Import required libraries
from smolagents import CodeAgent, HfApiModel, tool
import gradio as gr
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import search tool
from smolagents import DuckDuckGoSearchTool

# Custom tool: Convert Celsius to Fahrenheit
@tool
def celsius_to_fahrenheit(celsius: float) -> float:
    """
    Convert temperature from Celsius to Fahrenheit.
    
    Args:
        celsius (float): Temperature in Celsius
        
    Returns:
        float: Temperature in Fahrenheit
    """
    return (celsius * 9/5) + 32

# Get Hugging Face token from environment variable
hf_token = os.getenv("HF_TOKEN")

# Check if token exists
if not hf_token:
    raise ValueError("⚠️ HF_TOKEN environment variable is not set! Please create a .env file with your token.")

# Initialize the model with auto provider
model = HfApiModel(
    provider="auto",  # Auto-select the best provider
    token=hf_token
)

# Create the Code Agent with tools
agent = CodeAgent(
    model=model,
    tools=[
        DuckDuckGoSearchTool(),      # Web search capability
        celsius_to_fahrenheit         # Custom temperature conversion
    ],
    max_steps=10,                     # Maximum reasoning steps
    verbosity_level=1                 # Show execution details
)

# Function to run the agent with user message
def run_agent(message, history):
    """
    Execute the agent with user input and return response.
    
    Args:
        message (str): User's question or command
        history: Chat history (managed by Gradio)
        
    Returns:
        str: Agent's response or error message
    """
    try:
        response = agent.run(message)
        return str(response)
    except Exception as e:
        return f"❌ Error: {str(e)}"

# Create Gradio Chat Interface
demo = gr.ChatInterface(
    fn=run_agent,
    title="🤖 My First AI Agent",
    description="Ask me anything! I can search the web and convert temperatures.",
    examples=[
        "What's the weather in Tehran?",
        "Convert 25 Celsius to Fahrenheit",
        "What are the latest AI news?",
        "Who won the last World Cup?"
    ],
    theme="soft"
)

# Launch the application
if __name__ == "__main__":
    demo.launch()
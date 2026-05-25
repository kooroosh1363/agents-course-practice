# Import required libraries
from smolagents import CodeAgent, HfApiModel, tool
import gradio as gr
import os
from dotenv import load_dotenv
import secrets
import string

# Load environment variables from .env file
load_dotenv()

# Import web search tool
from smolagents import DuckDuckGoSearchTool

# Tool 1: Calculate percentage
@tool
def calculate_percentage(part: float, whole: float) -> float:
    """
    Calculate what percentage 'part' is of 'whole'.
    
    Args:
        part (float): The part value
        whole (float): The whole value
        
    Returns:
        float: The calculated percentage
    """
    if whole == 0:
        raise ValueError("Whole value cannot be zero.")
    return (part / whole) * 100

# Tool 2: Convert bytes to human-readable format
@tool
def convert_bytes(bytes_val: int) -> str:
    """
    Convert bytes to KB, MB, GB, or TB.
    
    Args:
        bytes_val (int): Size in bytes
        
    Returns:
        str: Human-readable size string
    """
    if bytes_val < 0:
        raise ValueError("Bytes value cannot be negative.")
    units = ['B', 'KB', 'MB', 'GB', 'TB']
    unit_index = 0
    size = float(bytes_val)
    while size >= 1024.0 and unit_index < len(units) - 1:
        size /= 1024.0
        unit_index += 1
    return f"{size:.2f} {units[unit_index]}"

# Tool 3: Generate secure password
@tool
def generate_secure_password(length: int) -> str:
    """
    Generate a cryptographically secure random password.
    
    Args:
        length (int): Desired password length (min 8, max 64)
        
    Returns:
        str: Generated password
    """
    length = max(8, min(length, 64))
    alphabet = string.ascii_letters + string.digits + string.punctuation
    return ''.join(secrets.choice(alphabet) for _ in range(length))

# Get Hugging Face token from environment variable
hf_token = os.getenv("HF_TOKEN")

# Validate token existence
if not hf_token:
    raise ValueError("⚠️ HF_TOKEN environment variable is not set! Please create a .env file with your token.")

# Initialize the model with auto provider
model = HfApiModel(
    provider="auto",
    token=hf_token
)

# Create the Code Agent with utility tools
agent = CodeAgent(
    model=model,
    tools=[
        DuckDuckGoSearchTool(),      # Web search capability
        calculate_percentage,         # Math utility
        convert_bytes,                # File size converter
        generate_secure_password      # Password generator
    ],
    max_steps=10,
    verbosity_level=1
)

# Function to run the agent with user message
def run_agent(message, history):
    """
    Execute the agent with user input and return response.
    """
    try:
        response = agent.run(message)
        return str(response)
    except Exception as e:
        return f" Error: {str(e)}"

# Create Gradio Chat Interface
demo = gr.ChatInterface(
    fn=run_agent,
    title="️ Developer Utility Agent",
    description="I can calculate percentages, convert file sizes, generate secure passwords, and search the web for tech info.",
    examples=[
        "What percentage is 45 of 200?",
        "Convert 15728640 bytes to MB",
        "Generate a secure 16-character password",
        "How to fix 'ModuleNotFoundError' in Python?"
    ],
    theme="default"
)

# Launch the application
if __name__ == "__main__":
    demo.launch()
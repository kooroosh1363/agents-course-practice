# Import required libraries
from smolagents import CodeAgent, HfApiModel, tool
import gradio as gr
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import web search tool
from smolagents import DuckDuckGoSearchTool

# Tool 1: Estimate reading time based on word count
@tool
def estimate_reading_time(word_count: int) -> float:
    """
    Estimate reading time in minutes (average speed: 200 words/min).
    
    Args:
        word_count (int): Total number of words
        
    Returns:
        float: Estimated reading time in minutes
    """
    if word_count < 0:
        raise ValueError("Word count cannot be negative.")
    return round(word_count / 200, 2)

# Tool 2: Calculate Pomodoro sessions & breaks
@tool
def calculate_pomodoro_sessions(total_minutes: int) -> str:
    """
    Plan Pomodoro technique schedule (25 min work + 5 min break).
    
    Args:
        total_minutes (int): Available time in minutes
        
    Returns:
        str: Formatted schedule breakdown
    """
    if total_minutes <= 0:
        raise ValueError("Time must be positive.")
    session_duration = 30  # 25 work + 5 break
    full_sessions = total_minutes // session_duration
    remaining = total_minutes % session_duration
    work_time = full_sessions * 25
    break_time = full_sessions * 5
    return (f"✅ {full_sessions} full Pomodoro sessions\n"
            f"⏱️ Total work time: {work_time} mins\n"
            f"☕ Total break time: {break_time} mins\n"
            f"📌 Remaining time: {remaining} mins")

# Tool 3: Calculate compound interest
@tool
def calculate_compound_interest(principal: float, annual_rate: float, years: int) -> float:
    """
    Calculate future value with annual compound interest.
    
    Args:
        principal (float): Initial investment amount
        annual_rate (float): Annual interest rate percentage (e.g., 5 for 5%)
        years (int): Investment duration in years
        
    Returns:
        float: Final amount after compound interest
    """
    if principal < 0 or annual_rate < 0 or years < 0:
        raise ValueError("All inputs must be non-negative.")
    rate_decimal = annual_rate / 100
    return round(principal * (1 + rate_decimal) ** years, 2)

# Get Hugging Face token
hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    raise ValueError("⚠️ HF_TOKEN environment variable is not set! Please create a .env file with your token.")

# Initialize model
model = HfApiModel(
    provider="auto",
    token=hf_token
)

# Create the Code Agent with productivity & finance tools
agent = CodeAgent(
    model=model,
    tools=[
        DuckDuckGoSearchTool(),      # Web search
        estimate_reading_time,        # Reading planner
        calculate_pomodoro_sessions,  # Time management
        calculate_compound_interest   # Financial calculator
    ],
    max_steps=10,
    verbosity_level=1
)

# Function to run the agent
def run_agent(message, history):
    """
    Execute agent and return response.
    """
    try:
        response = agent.run(message)
        return str(response)
    except Exception as e:
        return f"❌ Error: {str(e)}"

# Create Gradio Chat Interface
demo = gr.ChatInterface(
    fn=run_agent,
    title=" Productivity & Finance Agent",
    description="I help with reading estimates, Pomodoro planning, compound interest calculations, and web search.",
    examples=[
        "How long to read a 2500-word article?",
        "Plan Pomodoro sessions for 120 minutes",
        "Calculate compound interest on $1000 at 7% for 5 years",
        "Best productivity apps for developers 2024"
    ],
    theme="glass"
)

# Launch the application
if __name__ == "__main__":
    demo.launch()
# Import required libraries
from smolagents import CodeAgent, HfApiModel, tool
import gradio as gr
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import web search tool
from smolagents import DuckDuckGoSearchTool

# Tool 1: Calculate Body Mass Index (BMI)
@tool
def calculate_bmi(weight_kg: float, height_m: float) -> float:
    """
    Calculate Body Mass Index (BMI).
    
    Args:
        weight_kg (float): Weight in kilograms
        height_m (float): Height in meters
        
    Returns:
        float: BMI value
    """
    if height_m <= 0:
        raise ValueError("Height must be positive.")
    return round(weight_kg / (height_m ** 2), 2)

# Tool 2: Calculate recommended daily water intake
@tool
def daily_water_intake(weight_kg: float) -> float:
    """
    Calculate recommended daily water intake in liters (approx weight * 0.033).
    
    Args:
        weight_kg (float): Weight in kilograms
        
    Returns:
        float: Recommended water intake in liters
    """
    return round(weight_kg * 0.033, 2)

# Tool 3: Calculate estimated maximum heart rate
@tool
def calculate_max_heart_rate(age: int) -> int:
    """
    Calculate estimated maximum heart rate using the formula (220 - age).
    
    Args:
        age (int): Age in years
        
    Returns:
        int: Estimated maximum heart rate in beats per minute
    """
    return 220 - age

# Get Hugging Face token
hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    raise ValueError("⚠️ HF_TOKEN environment variable is not set! Please create a .env file with your token.")

# Initialize model
model = HfApiModel(
    provider="auto",
    token=hf_token
)

# Create the Code Agent with health tools
agent = CodeAgent(
    model=model,
    tools=[
        DuckDuckGoSearchTool(),      # Web search for health info
        calculate_bmi,                # BMI Calculator
        daily_water_intake,           # Hydration Calculator
        calculate_max_heart_rate      # Heart Rate Calculator
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
    title="🏃 Health & Fitness Agent",
    description="I can calculate BMI, recommend water intake, estimate max heart rate, and answer health questions.",
    examples=[
        "Calculate BMI for 70kg and 1.75m",
        "How much water should a 80kg person drink?",
        "What is the max heart rate for a 30 year old?",
        "Benefits of drinking water in the morning"
    ],
    theme="soft"
)

# Launch the application
if __name__ == "__main__":
    demo.launch()
# Import required libraries
from smolagents import CodeAgent, HfApiModel, tool
import gradio as gr
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import web search tool
from smolagents import DuckDuckGoSearchTool

# Tool 1: Scale recipe ingredients based on servings
@tool
def scale_recipe(amount: float, original_servings: int, target_servings: int) -> float:
    """
    Scale an ingredient amount based on desired number of servings.
    
    Args:
        amount (float): Original ingredient amount
        original_servings (int): Original recipe servings
        target_servings (int): Desired number of servings
        
    Returns:
        float: Scaled ingredient amount
    """
    if original_servings <= 0:
        raise ValueError("Original servings must be positive.")
    return round(amount * (target_servings / original_servings), 2)

# Tool 2: Convert cups to milliliters
@tool
def convert_cups_to_ml(cups: float) -> float:
    """
    Convert volume from cups to milliliters (1 cup = 236.588 ml).
    
    Args:
        cups (float): Volume in cups
        
    Returns:
        float: Volume in milliliters
    """
    if cups < 0:
        raise ValueError("Volume cannot be negative.")
    return round(cups * 236.588, 2)

# Tool 3: Calculate calories per serving
@tool
def calories_per_serving(total_calories: float, servings: int) -> float:
    """
    Calculate calories per single serving.
    
    Args:
        total_calories (float): Total calories in the dish
        servings (int): Number of servings
        
    Returns:
        float: Calories per serving
    """
    if servings <= 0:
        raise ValueError("Servings must be positive.")
    return round(total_calories / servings, 2)

# Get Hugging Face token
hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    raise ValueError("⚠️ HF_TOKEN environment variable is not set! Please create a .env file with your token.")

# Initialize model
model = HfApiModel(
    provider="auto",
    token=hf_token
)

# Create the Code Agent with cooking tools
agent = CodeAgent(
    model=model,
    tools=[
        DuckDuckGoSearchTool(),      # Web search for recipes & tips
        scale_recipe,                 # Recipe scaler
        convert_cups_to_ml,           # Volume converter
        calories_per_serving          # Nutrition calculator
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
    title=" Cooking & Nutrition Agent",
    description="I can scale recipes, convert cups to ml, calculate calories per serving, and find cooking tips.",
    examples=[
        "Scale 2 cups of flour for 4 servings to 6 servings",
        "Convert 1.5 cups to milliliters",
        "Calculate calories per serving for a 1200 calorie dish with 4 portions",
        "How to make authentic Persian Ghormeh Sabzi?"
    ],
    theme="soft"
)

# Launch the application
if __name__ == "__main__":
    demo.launch()
# Import required libraries
from smolagents import CodeAgent, HfApiModel, tool
import gradio as gr
import os
import math
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import web search tool
from smolagents import DuckDuckGoSearchTool

# Tool 1: Calculate CO2 emissions saved by walking/cycling instead of driving
@tool
def calculate_co2_saved_km(distance_km: float) -> float:
    """
    Calculate CO2 emissions saved by walking/cycling instead of driving.
    Assumes average car emits ~0.12 kg CO2 per km.
    
    Args:
        distance_km (float): Distance in kilometers
        
    Returns:
        float: CO2 saved in kilograms
    """
    if distance_km < 0:
        raise ValueError("Distance cannot be negative.")
    avg_car_emission_per_km = 0.12
    return round(distance_km * avg_car_emission_per_km, 2)

# Tool 2: Estimate water savings from reducing shower time
@tool
def estimate_water_saved_minutes(minutes_reduced: int) -> float:
    """
    Estimate water saved by reducing shower time.
    Assumes standard showerhead uses ~10 liters per minute.
    
    Args:
        minutes_reduced (int): Minutes reduced per shower
        
    Returns:
        float: Water saved in liters
    """
    if minutes_reduced < 0:
        raise ValueError("Minutes reduced cannot be negative.")
    liters_per_minute = 10.0
    return round(minutes_reduced * liters_per_minute, 2)

# Tool 3: Calculate number of trees needed to offset annual CO2
@tool
def calculate_trees_for_offset(annual_co2_kg: float) -> int:
    """
    Calculate how many mature trees are needed to offset annual CO2 emissions.
    Assumes one mature tree absorbs ~22 kg CO2 per year.
    
    Args:
        annual_co2_kg (float): Annual CO2 emissions in kilograms
        
    Returns:
        int: Number of trees needed (rounded up)
    """
    if annual_co2_kg < 0:
        raise ValueError("CO2 emissions cannot be negative.")
    co2_per_tree_per_year = 22.0
    return math.ceil(annual_co2_kg / co2_per_tree_per_year)

# Get Hugging Face token
hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    raise ValueError("⚠️ HF_TOKEN environment variable is not set! Please create a .env file with your token.")

# Initialize model
model = HfApiModel(
    provider="auto",
    token=hf_token
)

# Create the Code Agent with sustainability tools
agent = CodeAgent(
    model=model,
    tools=[
        DuckDuckGoSearchTool(),       # Web search for eco-tips
        calculate_co2_saved_km,        # Carbon footprint saver
        estimate_water_saved_minutes,  # Water conservation estimator
        calculate_trees_for_offset     # Reforestation calculator
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
    title="🌱 Sustainability & Eco Agent",
    description="I can calculate CO2 savings, estimate water conservation, plan tree planting offsets, and find eco-friendly tips.",
    examples=[
        "How much CO2 is saved by cycling 15 km instead of driving?",
        "Water saved if I reduce my shower by 3 minutes daily",
        "How many trees to offset 2000 kg of annual CO2?",
        "Best ways to reduce plastic waste at home"
    ],
    theme="soft"
)

# Launch the application
if __name__ == "__main__":
    demo.launch()
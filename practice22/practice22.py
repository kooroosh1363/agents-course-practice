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

# Tool 1: Calculate Molarity (M = moles / volume in L)
@tool
def calculate_molarity(moles: float, volume_liters: float) -> float:
    """
    Calculate molar concentration of a solution.
    
    Args:
        moles (float): Amount of solute in moles
        volume_liters (float): Solution volume in liters
        
    Returns:
        float: Molarity (mol/L)
    """
    if volume_liters <= 0:
        raise ValueError("Volume must be positive.")
    return round(moles / volume_liters, 4)

# Tool 2: Calculate water needed for dilution (C1V1 = C2V2)
@tool
def calculate_dilution_water_needed(c1: float, v1: float, c2: float) -> float:
    """
    Calculate volume of water to add for desired dilution.
    Returns water volume in the same unit as v1.
    
    Args:
        c1 (float): Initial concentration
        v1 (float): Initial volume
        c2 (float): Target concentration (must be < c1)
        
    Returns:
        float: Water volume to add
    """
    if c1 <= 0 or v1 <= 0 or c2 <= 0:
        raise ValueError("All inputs must be positive.")
    if c2 >= c1:
        raise ValueError("Target concentration must be lower than initial concentration.")
    v2 = (c1 * v1) / c2
    return round(v2 - v1, 4)

# Tool 3: Calculate pH from H+ concentration
@tool
def calculate_ph(h_concentration: float) -> float:
    """
    Calculate pH value from hydrogen ion concentration in mol/L.
    pH = -log10[H+]
    
    Args:
        h_concentration (float): H+ concentration in mol/L
        
    Returns:
        float: pH value
    """
    if h_concentration <= 0:
        raise ValueError("Concentration must be positive.")
    return round(-math.log10(h_concentration), 2)

# Tool 4: Convert Celsius to Kelvin
@tool
def convert_celsius_to_kelvin(celsius: float) -> float:
    """
    Convert temperature from Celsius to Kelvin.
    
    Args:
        celsius (float): Temperature in Celsius
        
    Returns:
        float: Temperature in Kelvin
    """
    if celsius < -273.15:
        raise ValueError("Temperature cannot be below absolute zero.")
    return round(celsius + 273.15, 2)

# Get Hugging Face token
hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    raise ValueError("⚠️ HF_TOKEN environment variable is not set! Please create a .env file with your token.")

# Initialize model
model = HfApiModel(
    provider="auto",
    token=hf_token
)

# Create the Code Agent with chemistry tools
agent = CodeAgent(
    model=model,
    tools=[
        DuckDuckGoSearchTool(),          # Chemical data, safety sheets, reaction info
        calculate_molarity,               # Concentration calculator
        calculate_dilution_water_needed,  # Solution preparation helper
        calculate_ph,                     # Acidity/alkalinity analyzer
        convert_celsius_to_kelvin         # Temperature unit converter
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
    title=" Chemistry & Laboratory Agent",
    description="I can calculate molarity, plan dilutions, compute pH, convert temperatures, and search chemical databases.",
    examples=[
        "Molarity for 0.5 moles in 2 liters",
        "Water to add to dilute 100ml of 2M solution to 0.5M",
        "pH for H+ concentration of 0.001 mol/L",
        "Convert 25 Celsius to Kelvin",
        "Safety precautions for handling sulfuric acid"
    ],
    theme="default"
)

# Launch the application
if __name__ == "__main__":
    demo.launch()
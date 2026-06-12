# Import required libraries
from smolagents import CodeAgent, HfApiModel, tool
import gradio as gr
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import web search tool
from smolagents import DuckDuckGoSearchTool

# Tool 1: Convert Horsepower to Kilowatts
@tool
def convert_hp_to_kw(horsepower: float) -> float:
    """
    Convert engine power from Horsepower (HP) to Kilowatts (kW).
    
    Args:
        horsepower (float): Engine power in HP
        
    Returns:
        float: Power in kW
    """
    if horsepower < 0:
        raise ValueError("Horsepower cannot be negative.")
    return round(horsepower * 0.7457, 2)

# Tool 2: Calculate maximum driving range on a full tank
@tool
def calculate_fuel_range(tank_capacity_liters: float, fuel_consumption_l_per_100km: float) -> float:
    """
    Calculate estimated driving range on a full tank.
    
    Args:
        tank_capacity_liters (float): Fuel tank capacity in liters
        fuel_consumption_l_per_100km (float): Average consumption in L/100km
        
    Returns:
        float: Estimated range in kilometers
    """
    if tank_capacity_liters <= 0 or fuel_consumption_l_per_100km <= 0:
        raise ValueError("Tank capacity and consumption must be positive.")
    return round((tank_capacity_liters / fuel_consumption_l_per_100km) * 100, 2)

# Tool 3: Calculate fuel cost per kilometer
@tool
def calculate_cost_per_km(fuel_consumption_l_per_100km: float, price_per_liter: float) -> float:
    """
    Calculate fuel cost for driving 1 kilometer.
    
    Args:
        fuel_consumption_l_per_100km (float): Average consumption in L/100km
        price_per_liter (float): Fuel price per liter
        
    Returns:
        float: Cost per kilometer
    """
    if fuel_consumption_l_per_100km <= 0 or price_per_liter <= 0:
        raise ValueError("Inputs must be positive.")
    return round((fuel_consumption_l_per_100km / 100) * price_per_liter, 4)

# Tool 4: Calculate Power-to-Weight Ratio
@tool
def calculate_power_to_weight_ratio(horsepower: float, weight_kg: float) -> float:
    """
    Calculate power-to-weight ratio (HP per kg).
    Lower number usually indicates better acceleration performance.
    
    Args:
        horsepower (float): Engine power in HP
        weight_kg (float): Vehicle weight in kg
        
    Returns:
        float: HP per kg ratio
    """
    if weight_kg <= 0:
        raise ValueError("Weight must be positive.")
    return round(horsepower / weight_kg, 4)

# Get Hugging Face token
hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    raise ValueError("⚠️ HF_TOKEN environment variable is not set! Please create a .env file with your token.")

# Initialize model
model = HfApiModel(
    provider="auto",
    token=hf_token
)

# Create the Code Agent with automotive tools
agent = CodeAgent(
    model=model,
    tools=[
        DuckDuckGoSearchTool(),        # Car reviews, specs, maintenance tips
        convert_hp_to_kw,              # Power unit converter
        calculate_fuel_range,          # Range estimator
        calculate_cost_per_km,         # Running cost calculator
        calculate_power_to_weight_ratio # Performance metric
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
        return f" Error: {str(e)}"

# Create Gradio Chat Interface
demo = gr.ChatInterface(
    fn=run_agent,
    title="🚗 Automotive & Car Enthusiast Agent",
    description="I can convert HP to kW, estimate fuel range, calculate cost per km, and analyze power-to-weight ratios.",
    examples=[
        "Convert 300 HP to kW",
        "Range for 60L tank with 8L/100km consumption",
        "Cost per km if consumption is 10L/100km and gas is $1.5/L",
        "Power-to-weight ratio for 400HP car weighing 1600kg",
        "Best 2024 electric SUVs under $50k"
    ],
    theme="default"
)

# Launch the application
if __name__ == "__main__":
    demo.launch()
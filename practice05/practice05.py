# Import required libraries
from smolagents import CodeAgent, HfApiModel, tool
import gradio as gr
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import web search tool
from smolagents import DuckDuckGoSearchTool

# Tool 1: Convert kilometers to miles
@tool
def convert_km_to_miles(km: float) -> float:
    """
    Convert distance from kilometers to miles.
    
    Args:
        km (float): Distance in kilometers
        
    Returns:
        float: Distance in miles
    """
    if km < 0:
        raise ValueError("Distance cannot be negative.")
    return round(km * 0.621371, 2)

# Tool 2: Calculate estimated fuel cost for a trip
@tool
def calculate_fuel_cost(distance_km: float, consumption_l_per_100km: float, price_per_liter: float) -> float:
    """
    Calculate total fuel cost for a given distance.
    
    Args:
        distance_km (float): Total distance in kilometers
        consumption_l_per_100km (float): Fuel consumption in liters per 100 km
        price_per_liter (float): Fuel price per liter
        
    Returns:
        float: Total fuel cost
    """
    if any(x < 0 for x in [distance_km, consumption_l_per_100km, price_per_liter]):
        raise ValueError("All inputs must be non-negative.")
    liters_needed = (distance_km / 100) * consumption_l_per_100km
    return round(liters_needed * price_per_liter, 2)

# Tool 3: Estimate travel time
@tool
def estimate_travel_time(distance_km: float, avg_speed_kmh: float) -> float:
    """
    Estimate travel time in hours based on distance and average speed.
    
    Args:
        distance_km (float): Distance in kilometers
        avg_speed_kmh (float): Average speed in km/h
        
    Returns:
        float: Travel time in hours
    """
    if distance_km < 0 or avg_speed_kmh <= 0:
        raise ValueError("Distance must be non-negative and speed must be positive.")
    return round(distance_km / avg_speed_kmh, 2)

# Get Hugging Face token
hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    raise ValueError("⚠️ HF_TOKEN environment variable is not set! Please create a .env file with your token.")

# Initialize model
model = HfApiModel(
    provider="auto",
    token=hf_token
)

# Create the Code Agent with travel tools
agent = CodeAgent(
    model=model,
    tools=[
        DuckDuckGoSearchTool(),      # Web search for travel info
        convert_km_to_miles,          # Distance conversion
        calculate_fuel_cost,          # Fuel cost calculator
        estimate_travel_time          # Time estimator
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
    title="🌍 Travel & Logistics Agent",
    description="I can convert distances, calculate fuel costs, estimate travel time, and search for travel tips.",
    examples=[
        "Convert 500 km to miles",
        "Calculate fuel cost for 300 km trip, car consumes 7L/100km, fuel price is $1.5/L",
        "How long to drive 450 km at 90 km/h?",
        "Best road trip routes in Iran 2024"
    ],
    theme="default"
)

# Launch the application
if __name__ == "__main__":
    demo.launch()
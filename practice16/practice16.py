# Import required libraries
from smolagents import CodeAgent, HfApiModel, tool
import gradio as gr
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import web search tool
from smolagents import DuckDuckGoSearchTool

# Tool 1: Calculate estimated flight time
@tool
def calculate_flight_time(distance_km: float, cruise_speed_kmh: float) -> float:
    """
    Calculate estimated flight time in hours.
    
    Args:
        distance_km (float): Flight distance in kilometers
        cruise_speed_kmh (float): Average cruise speed in km/h
        
    Returns:
        float: Flight time in hours
    """
    if distance_km <= 0 or cruise_speed_kmh <= 0:
        raise ValueError("Distance and speed must be positive.")
    return round(distance_km / cruise_speed_kmh, 2)

# Tool 2: Convert km/h to knots (nautical miles per hour)
@tool
def convert_kmh_to_knots(speed_kmh: float) -> float:
    """
    Convert speed from kilometers per hour to knots.
    
    Args:
        speed_kmh (float): Speed in km/h
        
    Returns:
        float: Speed in knots
    """
    if speed_kmh < 0:
        raise ValueError("Speed cannot be negative.")
    return round(speed_kmh * 0.539957, 2)

# Tool 3: Calculate total fuel required
@tool
def calculate_fuel_needed(flight_hours: float, burn_rate_kg_per_hour: float) -> float:
    """
    Calculate total fuel needed for the planned flight duration.
    
    Args:
        flight_hours (float): Planned flight duration in hours
        burn_rate_kg_per_hour (float): Fuel consumption rate in kg/h
        
    Returns:
        float: Total fuel required in kg
    """
    if flight_hours <= 0 or burn_rate_kg_per_hour <= 0:
        raise ValueError("Flight hours and burn rate must be positive.")
    return round(flight_hours * burn_rate_kg_per_hour, 2)

# Tool 4: Calculate arrival time in UTC
@tool
def calculate_arrival_utc(departure_utc: str, flight_hours: float) -> str:
    """
    Calculate arrival time in UTC given departure time and flight duration.
    
    Args:
        departure_utc (str): Departure time in HH:MM format (24h)
        flight_hours (float): Flight duration in hours
        
    Returns:
        str: Arrival time in HH:MM format (UTC)
    """
    try:
        dep_time = datetime.strptime(departure_utc, "%H:%M")
        arr_time = dep_time + timedelta(hours=flight_hours)
        return arr_time.strftime("%H:%M")
    except ValueError:
        raise ValueError("Departure time must be in HH:MM 24-hour format.")

# Get Hugging Face token
hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    raise ValueError("⚠️ HF_TOKEN environment variable is not set! Please create a .env file with your token.")

# Initialize model
model = HfApiModel(
    provider="auto",
    token=hf_token
)

# Create the Code Agent with aviation tools
agent = CodeAgent(
    model=model,
    tools=[
        DuckDuckGoSearchTool(),       # Flight rules, weather, airport info
        calculate_flight_time,         # Time estimator
        convert_kmh_to_knots,          # Speed unit converter
        calculate_fuel_needed,         # Fuel planner
        calculate_arrival_utc          # Schedule calculator
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
    title="✈️ Aviation & Flight Planning Agent",
    description="I can calculate flight time, convert speed to knots, estimate fuel needs, plan arrival times, and search aviation resources.",
    examples=[
        "Flight time for 850 km at 420 km/h cruise speed",
        "Convert 300 km/h to knots",
        "Fuel needed for 3.5 hours flight burning 120 kg/h",
        "Arrival UTC if departing at 14:30 and flying 2.75 hours",
        "VFR weather minimums for Class G airspace"
    ],
    theme="default"
)

# Launch the application
if __name__ == "__main__":
    demo.launch()
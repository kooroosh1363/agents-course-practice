# Import required libraries
from smolagents import CodeAgent, HfApiModel, tool
import gradio as gr
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import web search tool
from smolagents import DuckDuckGoSearchTool

# Tool 1: Calculate Wind Chill Index
@tool
def calculate_wind_chill(temp_c: float, wind_kmh: float) -> float:
    """
    Calculate perceived temperature (wind chill) in Celsius.
    Valid for temp <= 10°C and wind >= 4.8 km/h.
    Uses standard meteorological formula.

    Args:
        temp_c (float): Air temperature in Celsius
        wind_kmh (float): Wind speed in km/h

    Returns:
        float: Wind chill temperature in Celsius
    """
    if temp_c > 10 or wind_kmh < 4.8:
        return temp_c  # Formula not applicable outside range
    return round(13.12 + 0.6215 * temp_c - 11.37 * (wind_kmh ** 0.16) + 0.3965 * temp_c * (wind_kmh ** 0.16), 2)

# Tool 2: Calculate rainwater harvesting volume
@tool
def calculate_rainwater_liters(catchment_area_sqm: float, rainfall_mm: float) -> float:
    """
    Calculate liters of rainwater collected from a roof/catchment area.
    Note: 1 mm of rain over 1 m² equals 1 liter.

    Args:
        catchment_area_sqm (float): Area in square meters
        rainfall_mm (float): Rainfall depth in millimeters

    Returns:
        float: Collected water in liters
    """
    if catchment_area_sqm < 0 or rainfall_mm < 0:
        raise ValueError("Area and rainfall must be non-negative.")
    return round(catchment_area_sqm * rainfall_mm, 2)

# Tool 3: Convert UV Index to safe sun exposure time
@tool
def calculate_uv_exposure_minutes(uv_index: float, skin_protection_factor: float) -> float:
    """
    Estimate safe sun exposure time in minutes.
    Formula: (Skin Factor * 10) / UV Index
    Skin Factor: 1 (very fair) to 4 (dark)

    Args:
        uv_index (float): Current UV index
        skin_protection_factor (float): Skin type factor (1-4)

    Returns:
        float: Safe exposure time in minutes
    """
    if uv_index <= 0:
        raise ValueError("UV index must be positive.")
    return round((skin_protection_factor * 10) / uv_index, 2)

# Tool 4: Estimate altitude from atmospheric pressure
@tool
def estimate_altitude_from_pressure(pressure_hpa: float) -> float:
    """
    Estimate altitude in meters based on atmospheric pressure.
    Uses barometric formula approximation (Standard sea level = 1013.25 hPa).

    Args:
        pressure_hpa (float): Atmospheric pressure in hectopascals (hPa/mbar)

    Returns:
        float: Approximate altitude in meters
    """
    if pressure_hpa <= 0:
        raise ValueError("Pressure must be positive.")
    return round(44330 * (1 - (pressure_hpa / 1013.25) ** 0.1903), 2)

# Get Hugging Face token
hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    raise ValueError("⚠️ HF_TOKEN environment variable is not set! Please create a .env file with your token.")

# Initialize model
model = HfApiModel(
    provider="auto",
    token=hf_token
)

# Create the Code Agent with meteorology tools
agent = CodeAgent(
    model=model,
    tools=[
        DuckDuckGoSearchTool(),       # Weather forecasts, climate news
        calculate_wind_chill,          # Perceived temperature
        calculate_rainwater_liters,    # Water harvesting
        calculate_uv_exposure_minutes, # Sun safety
        estimate_altitude_from_pressure # Altitude estimator
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
    title="🌦️ Meteorology & Weather Agent",
    description="I can calculate wind chill, rainwater harvest volume, UV safe exposure time, estimate altitude from pressure, and search weather data.",
    examples=[
        "Wind chill for 2°C and 20 km/h wind",
        "Rainwater from 50 sqm roof with 15mm rain",
        "Safe sun time for UV index 8 and skin factor 2",
        "Altitude for 850 hPa pressure",
        "What causes El Niño?"
    ],
    theme="default"
)

# Launch the application
if __name__ == "__main__":
    demo.launch()
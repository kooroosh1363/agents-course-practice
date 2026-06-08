# Import required libraries
from smolagents import CodeAgent, HfApiModel, tool
import gradio as gr
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import web search tool
from smolagents import DuckDuckGoSearchTool

# Tool 1: Calculate irrigation water needed in liters
@tool
def calculate_irrigation_liters(area_sqm: float, evaporation_rate_mm: float) -> float:
    """
    Calculate irrigation water required in liters.
    Note: 1 mm of water over 1 m² equals 1 liter.
    
    Args:
        area_sqm (float): Field area in square meters
        evaporation_rate_mm (float): Daily evaporation/requirement in mm
        
    Returns:
        float: Water needed in liters
    """
    if area_sqm < 0 or evaporation_rate_mm < 0:
        raise ValueError("Area and evaporation rate must be non-negative.")
    return round(area_sqm * evaporation_rate_mm, 2)

# Tool 2: Estimate total crop yield in kg
@tool
def estimate_crop_yield_kg(area_hectares: float, avg_yield_kg_per_hectare: float) -> float:
    """
    Estimate total harvest yield based on area and average yield per hectare.
    
    Args:
        area_hectares (float): Field area in hectares
        avg_yield_kg_per_hectare (float): Expected yield per hectare in kg
        
    Returns:
        float: Total estimated yield in kg
    """
    if area_hectares < 0 or avg_yield_kg_per_hectare < 0:
        raise ValueError("Area and yield rate must be non-negative.")
    return round(area_hectares * avg_yield_kg_per_hectare, 2)

# Tool 3: Convert hectares to acres
@tool
def convert_hectares_to_acres(hectares: float) -> float:
    """
    Convert land area from hectares to acres.
    
    Args:
        hectares (float): Area in hectares
        
    Returns:
        float: Area in acres
    """
    if hectares < 0:
        raise ValueError("Area cannot be negative.")
    return round(hectares * 2.47105, 2)

# Tool 4: Calculate fertilizer quantity needed
@tool
def calculate_fertilizer_kg(area_hectares: float, recommended_kg_per_hectare: float) -> float:
    """
    Calculate total fertilizer required based on area and recommendation.
    
    Args:
        area_hectares (float): Field area in hectares
        recommended_kg_per_hectare (float): Fertilizer recommendation per hectare
        
    Returns:
        float: Total fertilizer needed in kg
    """
    if area_hectares < 0 or recommended_kg_per_hectare < 0:
        raise ValueError("Area and recommendation must be non-negative.")
    return round(area_hectares * recommended_kg_per_hectare, 2)

# Get Hugging Face token
hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    raise ValueError("⚠️ HF_TOKEN environment variable is not set! Please create a .env file with your token.")

# Initialize model
model = HfApiModel(
    provider="auto",
    token=hf_token
)

# Create the Code Agent with farming tools
agent = CodeAgent(
    model=model,
    tools=[
        DuckDuckGoSearchTool(),        # Web search for crop tips & weather
        calculate_irrigation_liters,     # Water management
        estimate_crop_yield_kg,          # Harvest estimator
        convert_hectares_to_acres,       # Land unit converter
        calculate_fertilizer_kg          # Input planning
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
    title="🌾 Agriculture & Smart Farming Agent",
    description="I can calculate irrigation needs, estimate crop yields, convert land units, plan fertilizer usage, and search for farming tips.",
    examples=[
        "Water needed for 5000 sqm field with 6mm daily evaporation",
        "Estimate yield for 10 hectares with 4500 kg/hectare average",
        "Convert 25 hectares to acres",
        "Fertilizer needed for 8 hectares at 200 kg/hectare rate",
        "Best drought-resistant wheat varieties 2024"
    ],
    theme="default"
)

# Launch the application
if __name__ == "__main__":
    demo.launch()
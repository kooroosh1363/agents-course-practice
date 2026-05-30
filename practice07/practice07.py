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

# Tool 1: Calculate paint needed for a room
@tool
def calculate_paint_needed(area_m2: float, coats: int) -> float:
    """
    Calculate liters of paint needed for a given area and number of coats.
    Assumes standard coverage: 1 liter covers 10 m² per coat.
    
    Args:
        area_m2 (float): Wall area in square meters
        coats (int): Number of paint coats
        
    Returns:
        float: Required paint in liters
    """
    if area_m2 <= 0 or coats <= 0:
        raise ValueError("Area and coats must be positive.")
    coverage_per_liter = 10.0
    return round((area_m2 * coats) / coverage_per_liter, 2)

# Tool 2: Convert square meters to square feet
@tool
def convert_sqm_to_sqft(area_m2: float) -> float:
    """
    Convert area from square meters to square feet.
    
    Args:
        area_m2 (float): Area in square meters
        
    Returns:
        float: Area in square feet
    """
    if area_m2 < 0:
        raise ValueError("Area cannot be negative.")
    return round(area_m2 * 10.764, 2)

# Tool 3: Calculate number of tiles needed
@tool
def calculate_tile_count(area_m2: float, tile_size_cm: float) -> int:
    """
    Calculate number of square tiles needed to cover an area.
    Automatically adds 10% extra for cuts and waste.
    
    Args:
        area_m2 (float): Floor/wall area in square meters
        tile_size_cm (float): Tile side length in centimeters
        
    Returns:
        int: Total tiles needed (including 10% waste)
    """
    if area_m2 <= 0 or tile_size_cm <= 0:
        raise ValueError("Area and tile size must be positive.")
    tile_area_m2 = (tile_size_cm / 100) ** 2
    base_tiles = area_m2 / tile_area_m2
    total_tiles = math.ceil(base_tiles * 1.10)  # 10% waste factor
    return int(total_tiles)

# Get Hugging Face token
hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    raise ValueError("⚠️ HF_TOKEN environment variable is not set! Please create a .env file with your token.")

# Initialize model
model = HfApiModel(
    provider="auto",
    token=hf_token
)

# Create the Code Agent with DIY tools
agent = CodeAgent(
    model=model,
    tools=[
        DuckDuckGoSearchTool(),      # Web search for DIY guides
        calculate_paint_needed,       # Paint estimator
        convert_sqm_to_sqft,          # Area converter
        calculate_tile_count          # Tile calculator
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
    title="🏠 Home Improvement & DIY Agent",
    description="I can calculate paint needed, convert area units, estimate tile counts, and search for renovation tips.",
    examples=[
        "How much paint for 45 m² walls with 2 coats?",
        "Convert 30 square meters to square feet",
        "How many 60cm tiles for a 20 m² floor?",
        "Best paint types for humid bathrooms"
    ],
    theme="default"
)

# Launch the application
if __name__ == "__main__":
    demo.launch()
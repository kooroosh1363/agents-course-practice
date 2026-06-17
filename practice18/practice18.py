# Import required libraries
from smolagents import CodeAgent, HfApiModel, tool
import gradio as gr
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import web search tool
from smolagents import DuckDuckGoSearchTool

# Tool 1: Calculate KDA Ratio (Kills + Assists) / Deaths
@tool
def calculate_kda(kills: int, deaths: int, assists: int) -> float:
    """
    Calculate KDA ratio for MOBA/FPS games.
    Formula: (Kills + Assists) / Deaths (if deaths = 0, return K+A)
    
    Args:
        kills (int): Number of kills
        deaths (int): Number of deaths
        assists (int): Number of assists
        
    Returns:
        float: KDA ratio
    """
    if deaths < 0 or kills < 0 or assists < 0:
        raise ValueError("Stats cannot be negative.")
    if deaths == 0:
        return float(kills + assists)
    return round((kills + assists) / deaths, 2)

# Tool 2: Calculate XP needed to reach next level
@tool
def calculate_xp_to_level(current_xp: int, xp_for_next_level: int) -> int:
    """
    Calculate remaining XP needed to reach the next level.
    
    Args:
        current_xp (int): Current XP amount
        xp_for_next_level (int): XP threshold for next level
        
    Returns:
        int: XP remaining to level up
    """
    if current_xp < 0 or xp_for_next_level <= 0:
        raise ValueError("XP values must be non-negative and threshold positive.")
    remaining = xp_for_next_level - current_xp
    return max(0, remaining)

# Tool 3: Convert match duration from minutes to HH:MM format
@tool
def format_match_duration(total_minutes: int) -> str:
    """
    Format match duration from total minutes to HH:MM string.
    
    Args:
        total_minutes (int): Match duration in minutes
        
    Returns:
        str: Formatted time string (e.g., "01:35")
    """
    if total_minutes < 0:
        raise ValueError("Duration cannot be negative.")
    hours = total_minutes // 60
    minutes = total_minutes % 60
    return f"{hours:02d}:{minutes:02d}"

# Tool 4: Calculate Damage Per Second (DPS)
@tool
def calculate_dps(total_damage: float, fight_duration_seconds: float) -> float:
    """
    Calculate average damage per second dealt in combat.
    
    Args:
        total_damage (float): Total damage dealt
        fight_duration_seconds (float): Duration of combat in seconds
        
    Returns:
        float: Average DPS
    """
    if fight_duration_seconds <= 0:
        raise ValueError("Duration must be positive.")
    return round(total_damage / fight_duration_seconds, 2)

# Get Hugging Face token
hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    raise ValueError("⚠️ HF_TOKEN environment variable is not set! Please create a .env file with your token.")

# Initialize model
model = HfApiModel(
    provider="auto",
    token=hf_token
)

# Create the Code Agent with gaming tools
agent = CodeAgent(
    model=model,
    tools=[
        DuckDuckGoSearchTool(),      # Search for game guides, patch notes, esports news
        calculate_kda,                # Performance stat calculator
        calculate_xp_to_level,        # Progress tracker
        format_match_duration,        # Time formatter
        calculate_dps                 # Combat efficiency analyzer
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
    title="🎮 Gaming & Esports Agent",
    description="I can calculate KDA ratios, track XP progress, format match times, analyze DPS, and search gaming news.",
    examples=[
        "KDA for 12 kills, 3 deaths, 8 assists",
        "XP needed if I have 4500 and next level is 6000",
        "Format 95 minutes match duration",
        "DPS for 15000 damage in 45 seconds",
        "Latest League of Legends patch notes"
    ],
    theme="default"
)

# Launch the application
if __name__ == "__main__":
    demo.launch()
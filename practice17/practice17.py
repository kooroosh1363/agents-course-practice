# Import required libraries
from smolagents import CodeAgent, HfApiModel, tool
import gradio as gr
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import web search tool
from smolagents import DuckDuckGoSearchTool

# Tool 1: Calculate Player Average (Points/Goals per game)
@tool
def calculate_player_avg(total_score: float, games_played: int) -> float:
    """
    Calculate the average score or goals per game for a player.
    
    Args:
        total_score (float): Total points/goals scored
        games_played (int): Total number of games played
        
    Returns:
        float: Average score per game
    """
    if games_played <= 0:
        raise ValueError("Games played must be positive.")
    return round(total_score / games_played, 2)

# Tool 2: Calculate Team Win Rate Percentage
@tool
def calculate_win_rate(wins: int, losses: int, draws: int) -> float:
    """
    Calculate the win rate percentage for a team.
    
    Args:
        wins (int): Number of wins
        losses (int): Number of losses
        draws (int): Number of draws
        
    Returns:
        float: Win rate percentage
    """
    total_games = wins + losses + draws
    if total_games <= 0:
        raise ValueError("Total games must be positive.")
    return round((wins / total_games) * 100, 2)

# Tool 3: Calculate Running Pace (Min per KM)
@tool
def calculate_running_pace(distance_km: float, time_minutes: float) -> float:
    """
    Calculate running pace in minutes per kilometer.
    
    Args:
        distance_km (float): Distance run in km
        time_minutes (float): Total time in minutes
        
    Returns:
        float: Pace in min/km
    """
    if distance_km <= 0:
        raise ValueError("Distance must be positive.")
    return round(time_minutes / distance_km, 2)

# Tool 4: Project Season Stats based on current average
@tool
def project_season_stats(current_avg: float, total_season_games: int, games_remaining: int) -> float:
    """
    Project total season score based on current average performance.
    
    Args:
        current_avg (float): Current average score per game
        total_season_games (int): Total games in the season
        games_remaining (int): Games left to play
        
    Returns:
        float: Projected total score for the season
    """
    if total_season_games <= games_remaining or games_remaining < 0:
        raise ValueError("Invalid game counts.")
    return round(current_avg * total_season_games, 2)

# Get Hugging Face token
hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    raise ValueError("⚠️ HF_TOKEN environment variable is not set! Please create a .env file with your token.")

# Initialize model
model = HfApiModel(
    provider="auto",
    token=hf_token
)

# Create the Code Agent with sports analysis tools
agent = CodeAgent(
    model=model,
    tools=[
        DuckDuckGoSearchTool(),      # Search for match results, player news
        calculate_player_avg,         # Performance calculator
        calculate_win_rate,           # Team stat analyzer
        calculate_running_pace,       # Fitness tracker
        project_season_stats          # Future projection
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
    title="⚽ Sports & Game Stats Agent",
    description="I can calculate player averages, team win rates, running pace, and project season stats.",
    examples=[
        "Calculate average for 45 goals in 20 games",
        "Win rate for 15 wins, 3 losses, 2 draws",
        "Pace for 10km run in 50 minutes",
        "Project total if avg is 25.5 and season is 82 games",
        "Latest NBA standings"
    ],
    theme="default"
)

# Launch the application
if __name__ == "__main__":
    demo.launch()
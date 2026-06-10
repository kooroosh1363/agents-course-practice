# Import required libraries
from smolagents import CodeAgent, HfApiModel, tool
import gradio as gr
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import web search tool
from smolagents import DuckDuckGoSearchTool

# Tool 1: Calculate delay effect time in milliseconds based on BPM
@tool
def calculate_delay_ms(bpm: float, beat_division: float) -> float:
    """
    Calculate delay time in ms for sync effects.
    beat_division: 1.0 = quarter note, 0.5 = eighth note, 0.25 = sixteenth note.
    
    Args:
        bpm (float): Project tempo in beats per minute
        beat_division (float): Note division multiplier (e.g., 0.5 for 1/8 note)
        
    Returns:
        float: Delay time in milliseconds
    """
    if bpm <= 0 or beat_division <= 0:
        raise ValueError("BPM and division must be positive.")
    ms_per_beat = 60000 / bpm
    return round(ms_per_beat * beat_division, 2)

# Tool 2: Estimate uncompressed WAV file size
@tool
def estimate_wav_size_mb(duration_sec: float, sample_rate: int, bit_depth: int, channels: int) -> float:
    """
    Calculate size of a single uncompressed WAV audio file.
    Formula: (seconds * sample_rate * bit_depth * channels) / 8 / 1024^2
    
    Args:
        duration_sec (float): Track duration in seconds
        sample_rate (int): Sample rate in Hz (e.g., 44100)
        bit_depth (int): Bit depth (e.g., 16 or 24)
        channels (int): Number of channels (1=mono, 2=stereo)
        
    Returns:
        float: File size in MB
    """
    if any(x <= 0 for x in [duration_sec, sample_rate, bit_depth, channels]):
        raise ValueError("All inputs must be positive.")
    bytes_per_sec = sample_rate * bit_depth * channels / 8
    total_bytes = duration_sec * bytes_per_sec
    return round(total_bytes / (1024 ** 2), 2)

# Tool 3: Convert seconds to MM:SS format string
@tool
def format_track_duration(total_seconds: float) -> str:
    """
    Format duration from seconds to readable MM:SS string.
    
    Args:
        total_seconds (float): Total duration in seconds
        
    Returns:
        str: Formatted time string (e.g., "03:45")
    """
    if total_seconds < 0:
        raise ValueError("Duration cannot be negative.")
    minutes = int(total_seconds // 60)
    seconds = int(total_seconds % 60)
    return f"{minutes:02d}:{seconds:02d}"

# Tool 4: Calculate total project storage estimate
@tool
def calculate_project_storage_mb(num_tracks: int, avg_size_mb: float) -> float:
    """
    Estimate total project storage for multiple audio tracks.
    
    Args:
        num_tracks (int): Number of tracks in project
        avg_size_mb (float): Average size per track in MB
        
    Returns:
        float: Total estimated storage in MB
    """
    if num_tracks < 0 or avg_size_mb < 0:
        raise ValueError("Inputs must be non-negative.")
    return round(num_tracks * avg_size_mb, 2)

# Get Hugging Face token
hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    raise ValueError("⚠️ HF_TOKEN environment variable is not set! Please create a .env file with your token.")

# Initialize model
model = HfApiModel(
    provider="auto",
    token=hf_token
)

# Create the Code Agent with music production tools
agent = CodeAgent(
    model=model,
    tools=[
        DuckDuckGoSearchTool(),       # Search for plugins, techniques, samples
        calculate_delay_ms,            # Sync delay calculator
        estimate_wav_size_mb,          # Audio file size estimator
        format_track_duration,         # Time formatter
        calculate_project_storage_mb   # Project storage planner
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
    title="🎵 Music & Audio Production Agent",
    description="I can calculate sync delay times, estimate audio file sizes, format durations, plan storage, and search for production tips.",
    examples=[
        "Delay time for 120 BPM on an eighth note (0.5)",
        "WAV size for 180 seconds, 44100Hz, 16bit, stereo",
        "Format 275 seconds to MM:SS",
        "Storage for 32 tracks averaging 45MB each",
        "Best free VST plugins for mixing 2024"
    ],
    theme="default"
)

# Launch the application
if __name__ == "__main__":
    demo.launch()
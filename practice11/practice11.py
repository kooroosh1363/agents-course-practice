# Import required libraries
from smolagents import CodeAgent, HfApiModel, tool
import gradio as gr
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import web search tool
from smolagents import DuckDuckGoSearchTool

# Tool 1: Calculate megapixels from image resolution
@tool
def calculate_megapixels(width: int, height: int) -> float:
    """
    Calculate image resolution in megapixels.
    
    Args:
        width (int): Image width in pixels
        height (int): Image height in pixels
        
    Returns:
        float: Resolution in megapixels
    """
    if width <= 0 or height <= 0:
        raise ValueError("Width and height must be positive integers.")
    return round((width * height) / 1_000_000, 2)

# Tool 2: Estimate storage required for a photo/video shoot
@tool
def estimate_storage_gb(num_photos: int, photo_size_mb: float, num_videos: int, video_size_mb: float) -> float:
    """
    Estimate total storage needed in gigabytes.
    
    Args:
        num_photos (int): Number of photos
        photo_size_mb (float): Average size per photo in MB
        num_videos (int): Number of videos
        video_size_mb (float): Average size per video in MB
        
    Returns:
        float: Total storage required in GB
    """
    if any(x < 0 for x in [num_photos, photo_size_mb, num_videos, video_size_mb]):
        raise ValueError("All inputs must be non-negative.")
    total_mb = (num_photos * photo_size_mb) + (num_videos * video_size_mb)
    return round(total_mb / 1024, 2)

# Tool 3: Calculate height to maintain aspect ratio
@tool
def calculate_height_for_aspect(width: int, ratio_w: int, ratio_h: int) -> int:
    """
    Calculate required height to maintain a specific aspect ratio.
    
    Args:
        width (int): Desired width in pixels
        ratio_w (int): Aspect ratio width (e.g., 16 for 16:9)
        ratio_h (int): Aspect ratio height (e.g., 9 for 16:9)
        
    Returns:
        int: Calculated height in pixels
    """
    if width <= 0 or ratio_w <= 0 or ratio_h <= 0:
        raise ValueError("Width and ratio values must be positive.")
    return round(width * ratio_h / ratio_w)

# Tool 4: Convert frame count to duration in seconds
@tool
def calculate_video_duration(frames: int, fps: int) -> float:
    """
    Calculate video duration in seconds from frame count and FPS.
    
    Args:
        frames (int): Total number of frames
        fps (int): Frames per second
        
    Returns:
        float: Duration in seconds
    """
    if frames < 0 or fps <= 0:
        raise ValueError("Frames must be non-negative and FPS must be positive.")
    return round(frames / fps, 2)

# Get Hugging Face token
hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    raise ValueError("⚠️ HF_TOKEN environment variable is not set! Please create a .env file with your token.")

# Initialize model
model = HfApiModel(
    provider="auto",
    token=hf_token
)

# Create the Code Agent with media tools
agent = CodeAgent(
    model=model,
    tools=[
        DuckDuckGoSearchTool(),      # Web search for photography tips/gear
        calculate_megapixels,         # Resolution calculator
        estimate_storage_gb,          # Storage estimator
        calculate_height_for_aspect,  # Aspect ratio helper
        calculate_video_duration      # Frame-to-time converter
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
    title="📸 Photography & Media Agent",
    description="I can calculate megapixels, estimate storage needs, maintain aspect ratios, convert frames to time, and search for media tips.",
    examples=[
        "Calculate megapixels for 6000x4000 resolution",
        "Storage needed for 500 photos (8MB each) and 10 videos (250MB each)",
        "What height for 1920 width in 16:9 ratio?",
        "Duration of 14400 frames at 60 FPS",
        "Best mirrorless cameras for beginners 2024"
    ],
    theme="default"
)

# Launch the application
if __name__ == "__main__":
    demo.launch()
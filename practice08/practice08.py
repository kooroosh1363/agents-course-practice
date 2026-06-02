# Import required libraries
from smolagents import CodeAgent, HfApiModel, tool
import gradio as gr
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import web search tool
from smolagents import DuckDuckGoSearchTool

# Tool 1: Calculate required final exam score
@tool
def calculate_required_final_score(current_avg: float, final_weight_pct: float, target_avg: float) -> float:
    """
    Calculate the score needed on the final exam to reach a target average.
    Formula: final_score = (target - current*(1-weight)) / weight
    
    Args:
        current_avg (float): Current course average
        final_weight_pct (float): Final exam weight percentage (0-100)
        target_avg (float): Desired final average
        
    Returns:
        float: Required score on final exam
    """
    w = final_weight_pct / 100.0
    if w <= 0 or w >= 1:
        raise ValueError("Final weight must be between 1 and 99 percent.")
    required = (target_avg - current_avg * (1 - w)) / w
    return round(required, 2)

# Tool 2: Estimate total study hours
@tool
def estimate_study_hours(total_pages: int, pages_per_hour: int) -> float:
    """
    Estimate total study hours based on material length and reading speed.
    
    Args:
        total_pages (int): Total number of pages to study
        pages_per_hour (int): Average reading/study speed in pages per hour
        
    Returns:
        float: Estimated hours needed
    """
    if pages_per_hour <= 0:
        raise ValueError("Study speed must be positive.")
    return round(total_pages / pages_per_hour, 2)

# Tool 3: Convert percentage to 4.0 GPA scale (approximate linear)
@tool
def convert_percentage_to_gpa(percentage: float) -> float:
    """
    Convert a percentage grade (0-100) to a 4.0 GPA scale.
    Note: This uses a simplified linear approximation for educational purposes.
    
    Args:
        percentage (float): Grade percentage (0-100)
        
    Returns:
        float: Approximate GPA on 4.0 scale
    """
    if not (0 <= percentage <= 100):
        raise ValueError("Percentage must be between 0 and 100.")
    return round(percentage / 25, 2)

# Tool 4: Calculate study progress percentage
@tool
def calculate_study_progress_percent(completed_tasks: int, total_tasks: int) -> float:
    """
    Calculate completion percentage for a study plan or checklist.
    
    Args:
        completed_tasks (int): Number of tasks completed
        total_tasks (int): Total number of tasks
        
    Returns:
        float: Progress percentage (0-100)
    """
    if total_tasks <= 0:
        raise ValueError("Total tasks must be positive.")
    return round((completed_tasks / total_tasks) * 100, 2)

# Get Hugging Face token
hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    raise ValueError("⚠️ HF_TOKEN environment variable is not set! Please create a .env file with your token.")

# Initialize model
model = HfApiModel(
    provider="auto",
    token=hf_token
)

# Create the Code Agent with education tools
agent = CodeAgent(
    model=model,
    tools=[
        DuckDuckGoSearchTool(),          # Web search for learning resources
        calculate_required_final_score,   # Grade planner
        estimate_study_hours,             # Time estimator
        convert_percentage_to_gpa,        # Grade converter
        calculate_study_progress_percent  # Progress tracker
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
    title="📚 Education & Learning Agent",
    description="I can calculate required final scores, estimate study hours, convert grades, and track study progress.",
    examples=[
        "Current avg 78, final worth 30%, target avg 85. What score do I need?",
        "How many hours to study 320 pages at 40 pages per hour?",
        "Convert 92% to GPA scale",
        "Progress percentage if 15 out of 24 tasks are done",
        "Best active recall techniques for exams"
    ],
    theme="soft"
)

# Launch the application
if __name__ == "__main__":
    demo.launch()
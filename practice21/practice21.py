# Import required libraries
from smolagents import CodeAgent, HfApiModel, tool
import gradio as gr
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import web search tool
from smolagents import DuckDuckGoSearchTool

# Tool 1: Calculate file download time in seconds
@tool
def calculate_download_time(file_size_mb: float, speed_mbps: float) -> float:
    """
    Calculate estimated download time in seconds.
    Note: 1 Byte = 8 bits. Speed is in Megabits per second.
    
    Args:
        file_size_mb (float): File size in Megabytes
        speed_mbps (float): Download speed in Megabits per second
        
    Returns:
        float: Download time in seconds
    """
    if file_size_mb <= 0 or speed_mbps <= 0:
        raise ValueError("File size and speed must be positive.")
    file_size_bits = file_size_mb * 8
    return round(file_size_bits / speed_mbps, 2)

# Tool 2: Calculate usable hosts in a subnet
@tool
def calculate_subnet_hosts(cidr_prefix: int) -> int:
    """
    Calculate the number of usable host addresses in a IPv4 subnet.
    Valid for CIDR prefixes /8 to /30.
    
    Args:
        cidr_prefix (int): Subnet mask prefix length (e.g., 24 for /24)
        
    Returns:
        int: Number of usable hosts
    """
    if cidr_prefix < 8 or cidr_prefix > 30:
        raise ValueError("CIDR prefix must be between 8 and 30 for standard subnets.")
    return (2 ** (32 - cidr_prefix)) - 2

# Tool 3: Convert bandwidth from Gbps to Mbps
@tool
def convert_gbps_to_mbps(gbps: float) -> float:
    """
    Convert network bandwidth from Gigabits per second to Megabits per second.
    
    Args:
        gbps (float): Bandwidth in Gbps
        
    Returns:
        float: Bandwidth in Mbps
    """
    if gbps < 0:
        raise ValueError("Bandwidth cannot be negative.")
    return round(gbps * 1000, 2)

# Tool 4: Calculate packet transmission time
@tool
def calculate_packet_transmission_time(packet_size_bits: float, bandwidth_bps: float) -> float:
    """
    Calculate time to transmit a single packet over a link.
    
    Args:
        packet_size_bits (float): Packet size in bits
        bandwidth_bps (float): Link bandwidth in bits per second
        
    Returns:
        float: Transmission time in seconds
    """
    if packet_size_bits <= 0 or bandwidth_bps <= 0:
        raise ValueError("Packet size and bandwidth must be positive.")
    return round(packet_size_bits / bandwidth_bps, 6)

# Get Hugging Face token
hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    raise ValueError("⚠️ HF_TOKEN environment variable is not set! Please create a .env file with your token.")

# Initialize model
model = HfApiModel(
    provider="auto",
    token=hf_token
)

# Create the Code Agent with networking tools
agent = CodeAgent(
    model=model,
    tools=[
        DuckDuckGoSearchTool(),          # RFC docs, protocol guides, troubleshooting
        calculate_download_time,           # Transfer estimator
        calculate_subnet_hosts,            # IP planning helper
        convert_gbps_to_mbps,              # Bandwidth unit converter
        calculate_packet_transmission_time # Link performance calculator
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
    title="📡 Network & IT Infrastructure Agent",
    description="I can calculate download times, subnet hosts, convert bandwidth units, analyze packet transmission, and search networking docs.",
    examples=[
        "Download time for 2.5 GB file at 100 Mbps",
        "Usable hosts in a /26 subnet",
        "Convert 10 Gbps to Mbps",
        "Transmission time for 1500 bytes packet at 1 Gbps",
        "Difference between TCP and UDP protocols"
    ],
    theme="default"
)

# Launch the application
if __name__ == "__main__":
    demo.launch()
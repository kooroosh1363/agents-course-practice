# Import required libraries
from smolagents import CodeAgent, HfApiModel, tool
import gradio as gr
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import web search tool
from smolagents import DuckDuckGoSearchTool

# Tool 1: Calculate fiat value of crypto holdings
@tool
def calculate_crypto_value(amount: float, price_per_coin: float) -> float:
    """
    Calculate the total fiat value of a cryptocurrency holding.
    
    Args:
        amount (float): Amount of cryptocurrency held
        price_per_coin (float): Current price per coin in fiat (e.g., USD)
        
    Returns:
        float: Total fiat value
    """
    if amount < 0 or price_per_coin < 0:
        raise ValueError("Amount and price cannot be negative.")
    return round(amount * price_per_coin, 2)

# Tool 2: Calculate Return on Investment (ROI) percentage
@tool
def calculate_roi(initial_investment: float, current_value: float) -> float:
    """
    Calculate ROI percentage for a crypto trade or investment.
    
    Args:
        initial_investment (float): Initial amount invested in fiat
        current_value (float): Current value of the investment in fiat
        
    Returns:
        float: ROI percentage
    """
    if initial_investment <= 0:
        raise ValueError("Initial investment must be positive.")
    return round(((current_value - initial_investment) / initial_investment) * 100, 2)

# Tool 3: Calculate trading fee
@tool
def calculate_trading_fee(trade_volume: float, fee_percentage: float) -> float:
    """
    Calculate the trading fee based on volume and exchange fee rate.
    
    Args:
        trade_volume (float): Total trade volume in fiat
        fee_percentage (float): Fee rate percentage (e.g., 0.1 for 0.1%)
        
    Returns:
        float: Trading fee amount
    """
    if trade_volume < 0 or fee_percentage < 0:
        raise ValueError("Volume and fee percentage must be non-negative.")
    return round(trade_volume * (fee_percentage / 100), 4)

# Tool 4: Calculate price per coin from market cap and supply
@tool
def calculate_price_from_marketcap(market_cap_usd: float, circulating_supply: float) -> float:
    """
    Derive the approximate price per coin using market cap and circulating supply.
    
    Args:
        market_cap_usd (float): Total market capitalization in USD
        circulating_supply (float): Number of coins in circulation
        
    Returns:
        float: Estimated price per coin in USD
    """
    if circulating_supply <= 0 or market_cap_usd < 0:
        raise ValueError("Market cap must be non-negative and supply must be positive.")
    return round(market_cap_usd / circulating_supply, 4)

# Get Hugging Face token
hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    raise ValueError("️ HF_TOKEN environment variable is not set! Please create a .env file with your token.")

# Initialize model
model = HfApiModel(
    provider="auto",
    token=hf_token
)

# Create the Code Agent with crypto tools
agent = CodeAgent(
    model=model,
    tools=[
        DuckDuckGoSearchTool(),      # Search for market news, coin data
        calculate_crypto_value,       # Portfolio valuator
        calculate_roi,                # Profit/loss analyzer
        calculate_trading_fee,        # Cost estimator
        calculate_price_from_marketcap # Fundamental metric calculator
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
    title="🪙 Cryptocurrency & Blockchain Agent",
    description="I can calculate portfolio value, ROI, trading fees, derive price from market cap, and search crypto news.",
    examples=[
        "Value of 2.5 ETH at $3200 per coin",
        "ROI if I invested $1000 and it's now $1450",
        "Trading fee for $5000 volume at 0.1% rate",
        "Price per coin if market cap is $1.2B and supply is 19M",
        "Latest Bitcoin halving date and impact"
    ],
    theme="default"
)

# Launch the application
if __name__ == "__main__":
    demo.launch()
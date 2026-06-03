# Import required libraries
from smolagents import CodeAgent, HfApiModel, tool
import gradio as gr
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import web search tool
from smolagents import DuckDuckGoSearchTool

# Tool 1: Calculate final price after discount
@tool
def calculate_discount(original_price: float, discount_percent: float) -> float:
    """
    Calculate the final price after applying a percentage discount.
    
    Args:
        original_price (float): Original price of the item
        discount_percent (float): Discount percentage (0-100)
        
    Returns:
        float: Final discounted price
    """
    if original_price < 0:
        raise ValueError("Price cannot be negative.")
    if not (0 <= discount_percent <= 100):
        raise ValueError("Discount must be between 0 and 100 percent.")
    discount_amount = original_price * (discount_percent / 100)
    return round(original_price - discount_amount, 2)

# Tool 2: Calculate price per unit (e.g., price per kg or per liter)
@tool
def calculate_price_per_unit(total_price: float, quantity: float) -> float:
    """
    Calculate the price per single unit (e.g., per kg, per liter, per item).
    
    Args:
        total_price (float): Total price paid
        quantity (float): Total quantity purchased
        
    Returns:
        float: Price per unit
    """
    if quantity <= 0:
        raise ValueError("Quantity must be positive.")
    return round(total_price / quantity, 2)

# Tool 3: Estimate shipping cost based on weight
@tool
def calculate_shipping_cost(weight_kg: float) -> float:
    """
    Estimate shipping cost based on weight.
    Standard rate: $5 base fee + $2 per kg.
    
    Args:
        weight_kg (float): Package weight in kilograms
        
    Returns:
        float: Estimated shipping cost
    """
    if weight_kg <= 0:
        raise ValueError("Weight must be positive.")
    base_fee = 5.0
    rate_per_kg = 2.0
    return round(base_fee + (weight_kg * rate_per_kg), 2)

# Get Hugging Face token
hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    raise ValueError("⚠️ HF_TOKEN environment variable is not set! Please create a .env file with your token.")

# Initialize model
model = HfApiModel(
    provider="auto",
    token=hf_token
)

# Create the Code Agent with shopping tools
agent = CodeAgent(
    model=model,
    tools=[
        DuckDuckGoSearchTool(),      # Web search for product reviews
        calculate_discount,           # Discount calculator
        calculate_price_per_unit,     # Unit price comparator
        calculate_shipping_cost       # Shipping estimator
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
    title="🛒 E-commerce & Shopping Agent",
    description="I can calculate discounts, compare unit prices, estimate shipping costs, and find product reviews.",
    examples=[
        "What is the final price of $150 item with 20% discount?",
        "Price per kg for a 5kg bag costing $40",
        "Shipping cost for a 12kg package",
        "Best mechanical keyboards under $100 reviews"
    ],
    theme="default"
)

# Launch the application
if __name__ == "__main__":
    demo.launch()
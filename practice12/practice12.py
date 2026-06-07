# Import required libraries
from smolagents import CodeAgent, HfApiModel, tool
import gradio as gr
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import web search tool
from smolagents import DuckDuckGoSearchTool

# Tool 1: Calculate monthly mortgage payment
@tool
def calculate_mortgage_payment(principal: float, annual_rate: float, years: int) -> float:
    """
    Calculate monthly mortgage payment using standard amortization formula.
    
    Args:
        principal (float): Loan amount
        annual_rate (float): Annual interest rate percentage (e.g., 4.5 for 4.5%)
        years (int): Loan term in years
        
    Returns:
        float: Monthly payment amount
    """
    if principal <= 0 or years <= 0 or annual_rate < 0:
        raise ValueError("Principal and years must be positive. Rate can be 0.")
    monthly_rate = annual_rate / 100 / 12
    num_payments = years * 12
    if monthly_rate == 0:
        return round(principal / num_payments, 2)
    payment = principal * (monthly_rate * (1 + monthly_rate)**num_payments) / ((1 + monthly_rate)**num_payments - 1)
    return round(payment, 2)

# Tool 2: Calculate price per square meter
@tool
def calculate_price_per_sqm(total_price: float, area_sqm: float) -> float:
    """
    Calculate property price per square meter.
    
    Args:
        total_price (float): Total property price
        area_sqm (float): Property area in square meters
        
    Returns:
        float: Price per square meter
    """
    if area_sqm <= 0:
        raise ValueError("Area must be positive.")
    return round(total_price / area_sqm, 2)

# Tool 3: Calculate annual rental yield percentage
@tool
def calculate_rental_yield_annual(annual_rent: float, property_price: float) -> float:
    """
    Calculate annual rental yield as a percentage.
    
    Args:
        annual_rent (float): Total rent collected per year
        property_price (float): Current property market value
        
    Returns:
        float: Rental yield percentage
    """
    if property_price <= 0:
        raise ValueError("Property price must be positive.")
    return round((annual_rent / property_price) * 100, 2)

# Tool 4: Check rent affordability (30% rule)
@tool
def check_rent_affordability(gross_monthly_income: float, monthly_rent: float) -> str:
    """
    Check if monthly rent fits within the standard 30% income rule.
    
    Args:
        gross_monthly_income (float): Monthly gross income
        monthly_rent (float): Proposed monthly rent
        
    Returns:
        str: Affordability status and threshold details
    """
    if gross_monthly_income <= 0:
        raise ValueError("Income must be positive.")
    threshold = gross_monthly_income * 0.30
    is_affordable = monthly_rent <= threshold
    status = "✅ Affordable" if is_affordable else "️ Exceeds 30% rule"
    return f"{status}. Max recommended: ${threshold:.2f}, Your rent: ${monthly_rent:.2f}"

# Get Hugging Face token
hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    raise ValueError("⚠️ HF_TOKEN environment variable is not set! Please create a .env file with your token.")

# Initialize model
model = HfApiModel(
    provider="auto",
    token=hf_token
)

# Create the Code Agent with real estate tools
agent = CodeAgent(
    model=model,
    tools=[
        DuckDuckGoSearchTool(),      # Market trends & neighborhood info
        calculate_mortgage_payment,   # Loan calculator
        calculate_price_per_sqm,      # Unit price comparator
        calculate_rental_yield_annual,# Investment return estimator
        check_rent_affordability      # Budget validator
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
    title="🏘️ Real Estate & Property Agent",
    description="I can calculate mortgage payments, price per sqm, rental yields, check rent affordability, and search market trends.",
    examples=[
        "Mortgage for $300,000 at 4.5% over 20 years",
        "Price per sqm for a $450,000 apartment with 120 sqm",
        "Rental yield if annual rent is $18,000 and property value is $300,000",
        "Is $1,200 rent affordable on $3,500 monthly income?",
        "Best neighborhoods for rental investment in Berlin 2024"
    ],
    theme="default"
)

# Launch the application
if __name__ == "__main__":
    demo.launch()
from fastmcp import FastMCP
import os

PORT = os.getenv("PORT", 8000)
mcp = FastMCP("demo",host=os.getenv("HOST", "0.0.0.0"),port=PORT)

@mcp.tool()
def add(a: float, b: float) -> float:
    """Add two numbers together.
    
    Args:
        a: First number to add
        b: Second number to add
        
    Returns:
        The sum of a and b
    """
    return a + b

@mcp.tool()
def multiply(a: float, b: float) -> float:
    """Multiply two numbers together.
    
    Args:
        a: First number to multiply
        b: Second number to multiply
        
    Returns:
        The product of a and b
    """
    return a * b

if __name__ == "__main__":
    mcp.run(transport="streamable-http")

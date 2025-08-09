from fastmcp import FastMCP

# Create MCP server instance
mcp = FastMCP("Math Operations Server")

# Export the app for Vercel
app = mcp.app

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

def main():
    """Run the MCP server."""
    print("Starting Math Operations MCP Server...")
    mcp.run()

if __name__ == "__main__":
    main()

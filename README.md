# Math Operations MCP Server

A simple Model Context Protocol (MCP) server built with FastMCP 2.0 that provides basic math operations.

## Features

This server provides two tools:
- **add**: Add two numbers together
- **multiply**: Multiply two numbers together

## Local Development

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the server locally:
```bash
python -c "from api.index import mcp; mcp.run(transport='http')"
```

Or create a simple local runner if needed:
```bash
cd api && python -c "import index; index.mcp.run(transport='http')"
```

## Deployment on Vercel

1. Install Vercel CLI:
```bash
npm i -g vercel
```

2. Deploy to Vercel:
```bash
vercel
```

3. Follow the prompts to configure your deployment.

## Usage

Once deployed, the MCP server will be available at your Vercel URL. The server exposes two tools:

### Add Tool
- **Function**: `add(a: float, b: float) -> float`
- **Description**: Adds two numbers together
- **Parameters**:
  - `a`: First number to add
  - `b`: Second number to add
- **Returns**: The sum of a and b

### Multiply Tool
- **Function**: `multiply(a: float, b: float) -> float`
- **Description**: Multiplies two numbers together
- **Parameters**:
  - `a`: First number to multiply
  - `b`: Second number to multiply
- **Returns**: The product of a and b

## Project Structure

```
MCPServer/
├── api/
│   ├── index.py        # MCP server implementation
│   └── requirements.txt # Python dependencies for Vercel
├── requirements.txt    # Python dependencies for local dev
├── vercel.json        # Vercel deployment configuration
├── pyproject.toml     # Project metadata
└── README.md          # This file
```

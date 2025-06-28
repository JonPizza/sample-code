import asyncio

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import create_tool_calling_agent
from langchain.agents import AgentExecutor
from langchain_openai import ChatOpenAI


# Create a tool using `@tool` wrapper
@tool
def add(a: float, b: float) -> float:
    """
    Add two numbers together.

    Args:
        a (float): First number
        b (float): Second number

    Returns:
        float: Sum of a and b
    """
    return a + b


# Get tools off of a remote MCP server
client = MultiServerMCPClient(
    {
        "coingecko": {
            "url": "https://mcp.api.coingecko.com/sse",
            "transport": "sse",  # <--- This will change depending on which server you use!
        }
    }
)


async def get_mcp_tools():
    tools = await client.get_tools()
    return tools


# Combine the tools into one list
tools = asyncio.run(get_mcp_tools())
tools.append(add)

# Reads the API key from environment variable `OPENAI_API_KEY` by default
# Use `gpt-4.1` since it is trained to do well with tool calls
llm = ChatOpenAI(model="gpt-4.1")

prompt = ChatPromptTemplate(
    [
        (
            "system",
            "You are a helpful assistant. Find the sum of the price of bitcoin and ethereum.",
        ),
        ("placeholder", "{messages}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)

tool_calling_agent = create_tool_calling_agent(llm, tools, prompt=prompt)
agent_executor = AgentExecutor(agent=tool_calling_agent, tools=tools, verbose=True)


async def generate_response(msg: str) -> str:
    response = await agent_executor.ainvoke({"messages": [HumanMessage(msg)]})
    return response["output"]


print(asyncio.run(generate_response("Can you sum the price of bitcoin and ethereum?")))

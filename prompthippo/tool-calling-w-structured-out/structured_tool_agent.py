import textwrap
import asyncio
from typing import Literal, List, Optional, Any

from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, ToolMessage, SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI


async def generate_structured_output(
    messages: List[BaseMessage],
    structure: BaseModel,
    tools: Optional[List[Any]] = None,
    model_name: Optional[str] = "gpt-4.1",
    temperature: float = 0.1,
    max_executions: int = 10,
) -> BaseModel:
    """
    Get a structured response from an LLM using LangChain's with_structured_output method
    with optional tool calling support.

    Args:
        messages: List of LangChain message objects
        structure: Pydantic BaseModel class defining the expected response structure
        tools: Optional list of LangChain tools that the model can use
        model_name: Specific model to use
        temperature: Model temperature for response generation
        max_executions:
            Maximum number of iterations for agent execution (if using tools).
            If `max_executions` = 1, then will simply run structured output.

    Returns:
        Instance of the provided Pydantic model with the LLM's response data
    """
    assert max_executions >= 1

    llm = ChatOpenAI(model=model_name, temperature=temperature)

    # If there are no tools, just simply run a LLM w/ structured output
    if not tools:
        model_with_structure = llm.with_structured_output(structure)
        result = await model_with_structure.ainvoke(messages)

        return result

    # Add the structure as a tool alongside other tools
    all_tools = tools + [structure]

    # Bind all tools including the structure as a tool
    # Use tool_choice="any" to force LLM to use any of the tools
    llm_with_tools = llm.bind_tools(all_tools, tool_choice="any")
    response = await llm_with_tools.ainvoke(messages)

    llm_forced_structure = llm.bind_tools([structure], tool_choice="any")

    for _ in range(max_executions - 1):
        structure_tool_call = None

        for tool_call in response.tool_calls:
            if tool_call["name"] == structure.__name__:
                structure_tool_call = tool_call
                break

        if structure_tool_call:
            if not structure_tool_call["args"]:
                raise Exception(
                    "Tool call result empty, likely ran out of tokens. Increase `max_tokens`."
                )

            # Create the structured response from the tool call arguments
            return structure(**structure_tool_call["args"])
        else:
            tool_messages = []

            for tool_call in response.tool_calls:
                # Find and execute the tool
                for tool in tools:
                    if hasattr(tool, "name") and tool.name == tool_call["name"]:
                        try:
                            tool_result = await tool.ainvoke(tool_call["args"])
                            tool_messages.append(
                                ToolMessage(
                                    content=str(tool_result),
                                    tool_call_id=tool_call["id"],
                                )
                            )
                            break
                        except Exception as e:
                            tool_messages.append(
                                ToolMessage(
                                    content=f"Error: {str(e)}",
                                    tool_call_id=tool_call["id"],
                                )
                            )
                            break

            # Add tool results to conversation and try again
            messages = messages + [response] + tool_messages
            response = await llm_with_tools.ainvoke(messages)

    # Reached max tool executions, generate structured response now.
    return await llm_forced_structure.ainvoke(messages)


class BookOutput(BaseModel):
    target_audience: Literal["kids", "teens", "young adult", "adult", "elderly"] = (
        Field(..., description="The target audience of the book")
    )
    reading_level: int = Field(
        ...,
        gt=0,
        lt=6,
        description="How difficult the book is to read for the target_audience. Higher is harder.",
    )


@tool
def collect_book_reviews(book_title: str) -> list[str]:
    """
    Retrieves book reviews from various trustworthy sources.

    Args:
        book_title (str): The title of the book to lookup reviews for

    Returns:
        list[str]: A list of book reviews
    """

    # A proper implementation that hits some API calls, etc. ...
    print(f'Called `collect_book_reviews` tool with book title "{book_title}"')

    return [
        "This book knocked my socks off! It was a difficult but managable read for me as I went into college.",
        "Tomorrow and Tomorrow and Tomorrow is a fantastic book for young adults who are looking to improve their reading level.",
    ]


async def main():
    return await generate_structured_output(
        [
            SystemMessage(
                textwrap.dedent("""\
                You are classifying books based on their first page of text.
                
                Use your `collect_book_reviews` tool.
                """)
            ),
            HumanMessage(
                textwrap.dedent("""\
                Tomorrow, and Tomorrow, and Tomorrow: Chapter 1

                Before Mazer invented himself as Mazer, he was Samson Mazer, and before he was Samson Mazer, he was Samson Masur—a change of two letters that transformed him from a nice, ostensibly Jewish boy to a Professional Builder of Worlds—and for most of his youth, he was Sam, S.A.M. on the hall of fame of his grandfather's Donkey Kong machine, but mainly Sam.

                On a late December afternoon, in the waning twentieth century, Sam exited a subway car and found the artery to the escalator clogged by an inert mass of people, who were gaping at a station advertisement. Sam was late. He had a meeting with his academic adviser that he had been postponing for over a month, but that everyone agreed absolutely needed to happen before winter break. Sam didn't care for crowds—being in them, or whatever foolishness they tended to enjoy en masse. But this crowd would not be avoided. He would have to force his way through it if he were to be delivered to the aboveground world.

                Sam wore an elephantine navy wool peacoat that he had inherited from his roommate, Marx, who had bought it freshman year from the Army Navy Surplus Store in town. Marx had left it moldering in its plastic shopping bag just short of an entire semester before Sam asked if he might borrow it. That winter had been unrelenting, and it was an April nor'easter (April! What madness, these Massachusetts winters!) that finally wore Sam's pride down enough to ask Marx for the forgotten coat. Sam pretended that he liked the style of it, and Marx said that Sam might as well take it, which is what Sam knew he would say. Like most things purchased from the Army Navy Surplus Store, the coat emanated mold, dust, and the perspiration of dead boys, and Sam tried not to speculate why the garment had been surplussed. But the coat was far warmer than the windbreaker he had brought from California his freshman year. He also believed that the large coat worked to conceal his size. The coat, its ridiculous scale, only made him look smaller and more childlike.
                """)
            ),
        ],
        BookOutput,
        [collect_book_reviews],
    )


if __name__ == "__main__":
    print(asyncio.run(main()))

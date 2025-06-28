import textwrap
from typing import Literal

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI


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


llm = ChatOpenAI(model="gpt-4.1")

llm_with_structure = llm.with_structured_output(BookOutput)

response = llm_with_structure.invoke(
    [
        (
            "system",
            "You are determining the target audience and reading level of a book based on the first page of the book.",
        ),
        (
            "human",
            textwrap.dedent("""\
        Tomorrow, and Tomorrow, and Tomorrow: Chapter 1

        Before Mazer invented himself as Mazer, he was Samson Mazer, and before he was Samson Mazer, he was Samson Masur—a change of two letters that transformed him from a nice, ostensibly Jewish boy to a Professional Builder of Worlds—and for most of his youth, he was Sam, S.A.M. on the hall of fame of his grandfather's Donkey Kong machine, but mainly Sam.

        On a late December afternoon, in the waning twentieth century, Sam exited a subway car and found the artery to the escalator clogged by an inert mass of people, who were gaping at a station advertisement. Sam was late. He had a meeting with his academic adviser that he had been postponing for over a month, but that everyone agreed absolutely needed to happen before winter break. Sam didn't care for crowds—being in them, or whatever foolishness they tended to enjoy en masse. But this crowd would not be avoided. He would have to force his way through it if he were to be delivered to the aboveground world.

        Sam wore an elephantine navy wool peacoat that he had inherited from his roommate, Marx, who had bought it freshman year from the Army Navy Surplus Store in town. Marx had left it moldering in its plastic shopping bag just short of an entire semester before Sam asked if he might borrow it. That winter had been unrelenting, and it was an April nor'easter (April! What madness, these Massachusetts winters!) that finally wore Sam's pride down enough to ask Marx for the forgotten coat. Sam pretended that he liked the style of it, and Marx said that Sam might as well take it, which is what Sam knew he would say. Like most things purchased from the Army Navy Surplus Store, the coat emanated mold, dust, and the perspiration of dead boys, and Sam tried not to speculate why the garment had been surplussed. But the coat was far warmer than the windbreaker he had brought from California his freshman year. He also believed that the large coat worked to conceal his size. The coat, its ridiculous scale, only made him look smaller and more childlike.
        """),
        ),
    ]
)

print(type(response))
print(response)

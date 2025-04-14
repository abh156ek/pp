# src/utils/state.py

from typing_extensions import Annotated, Sequence, TypedDict
import operator
from langchain_core.messages import BaseMessage


def merge_dicts(a: dict[str, any], b: dict[str, any]) -> dict[str, any]:
    """Merge two dictionaries, with values from b overwriting those from a."""
    return {**a, **b}


class AgentState(TypedDict):
    """Typed state object used by LangGraph agent nodes."""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    data: Annotated[dict[str, any], merge_dicts]
    metadata: Annotated[dict[str, any], merge_dicts]

# src/utils/state.py

class HumanMessage:
    def __init__(self, content, name):
        self.content = content
        self.name = name

# Keep the existing AgentState class here if you created it earlier
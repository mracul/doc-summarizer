from camel.agents import ChatAgent, CriticAgent
from camel.messages import BaseMessage
from rag_builder.prompts import (
    TOOL_CRITIC_AGENT_PROMPT,
    RETRIEVAL_AGENT_PROMPT,
    SYNTHESIS_AGENT_PROMPT,
    CLARIFICATION_AGENT_PROMPT,
)

class ToolCriticAgent(CriticAgent):
    """A critic agent that validates tool calls."""
    def __init__(self, *args, **kwargs):
        system_message = BaseMessage.make_assistant_message(
            role_name="Tool Critic",
            content=TOOL_CRITIC_AGENT_PROMPT,
        )
        super().__init__(system_message=system_message, *args, **kwargs)

class ClarificationAgent(ChatAgent):
    """An agent that clarifies a user's query and extracts search terms."""
    def __init__(self, *args, **kwargs):
        system_message = BaseMessage.make_assistant_message(
            role_name="Query Clarification Specialist",
            content=CLARIFICATION_AGENT_PROMPT,
        )
        super().__init__(system_message=system_message, tools=[], *args, **kwargs)

class RetrievalAgent(ChatAgent):
    """An agent that performs multi-hop retrieval."""
    def __init__(self, *args, **kwargs):
        system_message = BaseMessage.make_assistant_message(
            role_name="Retrieval Specialist",
            content=RETRIEVAL_AGENT_PROMPT,
        )
        super().__init__(system_message=system_message, *args, **kwargs)

class SynthesisAgent(ChatAgent):
    """An agent that synthesizes answers from retrieved context."""
    def __init__(self, *args, **kwargs):
        system_message = BaseMessage.make_assistant_message(
            role_name="Executive Synthesizer",
            content=SYNTHESIS_AGENT_PROMPT,
        )
        super().__init__(system_message=system_message, tools=[], *args, **kwargs)

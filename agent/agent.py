"""Minimal coding agent with tool use."""

from dataclasses import dataclass, field

from .llm import LLMClient
from .tools import TOOLS, run_tool
from .prompts import SYSTEM_PROMPT, format_task_prompt


@dataclass
class AgentResult:
    """Result of an agent run."""
    solution: str | None
    steps: int
    messages: list[dict] = field(default_factory=list)
    success: bool = False


class CodingAgent:
    """A minimal coding agent that solves programming tasks."""

    def __init__(self, model: str = "claude-sonnet-4-20250514", max_steps: int = 10):
        self.llm = LLMClient(model=model)
        self.max_steps = max_steps

    def solve(
        self,
        description: str,
        function_signature: str,
        examples: list[dict] | None = None,
    ) -> AgentResult:
        """
        Solve a coding task.

        Returns an AgentResult with the solution code (if found).
        """
        # Build initial message
        user_prompt = format_task_prompt(description, function_signature, examples)
        messages = [{"role": "user", "content": user_prompt}]

        solution = None
        steps = 0

        while steps < self.max_steps:
            steps += 1

            # Get LLM response
            response = self.llm.chat(
                messages=messages,
                system=SYSTEM_PROMPT,
                tools=TOOLS,
            )

            # Process response content
            assistant_content = []
            for block in response.content:
                if block.type == "text":
                    assistant_content.append({"type": "text", "text": block.text})
                elif block.type == "tool_use":
                    assistant_content.append({
                        "type": "tool_use",
                        "id": block.id,
                        "name": block.name,
                        "input": block.input,
                    })

            messages.append({"role": "assistant", "content": assistant_content})

            # Check for tool use
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    # Check if this is the final submission
                    if block.name == "submit_solution":
                        solution = block.input.get("code")
                        return AgentResult(
                            solution=solution,
                            steps=steps,
                            messages=messages,
                            success=True,
                        )

                    # Execute tool and collect result
                    result = run_tool(block.name, block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    })

            # If there were tool calls, add results and continue
            if tool_results:
                messages.append({"role": "user", "content": tool_results})
            else:
                # No tool calls and no submission - might be stuck
                break

            # Check stop reason
            if response.stop_reason == "end_turn" and not tool_results:
                break

        return AgentResult(
            solution=solution,
            steps=steps,
            messages=messages,
            success=False,
        )

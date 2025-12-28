"""Quick test to verify the agent works."""

from agent import CodingAgent


def main():
    agent = CodingAgent(max_steps=5)

    result = agent.solve(
        description="Write a function that returns the sum of two numbers.",
        function_signature="def add(a: int, b: int) -> int:",
        examples=[
            {"input": {"a": 2, "b": 3}, "expected": 5},
            {"input": {"a": -1, "b": 1}, "expected": 0},
        ],
    )

    print(f"Success: {result.success}")
    print(f"Steps: {result.steps}")
    print(f"Solution:\n{result.solution}")


if __name__ == "__main__":
    main()

"""LLM client wrapper supporting multiple providers."""

import os
import json
from dataclasses import dataclass
from typing import Any


@dataclass
class ToolUseBlock:
    """Represents a tool use request."""
    type: str
    id: str
    name: str
    input: dict


@dataclass
class TextBlock:
    """Represents a text response."""
    type: str
    text: str


@dataclass 
class LLMResponse:
    """Unified response format across providers."""
    content: list[Any]  # List of TextBlock or ToolUseBlock
    stop_reason: str


class LLMClient:
    """
    LLM client supporting multiple providers.
    
    Providers:
        - anthropic: Claude models (default)
        - groq: Llama, Mixtral (free tier available)
        - openai: GPT models
        - ollama: Local models
    
    Usage:
        # Anthropic (default)
        client = LLMClient(model="claude-sonnet-4-20250514")
        
        # Groq (free)
        client = LLMClient(model="llama-3.1-70b-versatile", provider="groq")
        
        # Ollama (local)
        client = LLMClient(model="llama3.1", provider="ollama")
    """
    
    # Default models per provider
    DEFAULT_MODELS = {
        "anthropic": "claude-sonnet-4-20250514",
        "groq": "llama-3.1-70b-versatile",
        "openai": "gpt-4o-mini",
        "ollama": "llama3.1",
    }

    def __init__(
        self, 
        model: str | None = None,
        provider: str | None = None,
    ):
        # If provider specified but not model, use default for that provider
        if provider and not model:
            model = self.DEFAULT_MODELS.get(provider, "claude-sonnet-4-20250514")
        elif not model:
            model = "claude-sonnet-4-20250514"
        
        self.model = model
        self.provider = provider or self._infer_provider(model)
        self._init_client()

    def _infer_provider(self, model: str) -> str:
        """Infer provider from model name."""
        if model.startswith("claude"):
            return "anthropic"
        elif model.startswith("gpt"):
            return "openai"
        elif model.startswith("llama") or model.startswith("mixtral"):
            return "groq"
        else:
            return "anthropic"  # default

    def _init_client(self):
        """Initialize the appropriate client."""
        if self.provider == "anthropic":
            import anthropic
            self.client = anthropic.Anthropic()
            
        elif self.provider == "groq":
            from groq import Groq
            self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
            
        elif self.provider == "openai":
            from openai import OpenAI
            self.client = OpenAI()
            
        elif self.provider == "ollama":
            # Ollama uses OpenAI-compatible API
            from openai import OpenAI
            self.client = OpenAI(
                base_url="http://localhost:11434/v1",
                api_key="ollama",  # not used but required
            )
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def chat(
        self,
        messages: list[dict],
        system: str = "",
        tools: list[dict] | None = None,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        """Send a chat request and return unified response."""
        
        if self.provider == "anthropic":
            return self._chat_anthropic(messages, system, tools, max_tokens)
        else:
            return self._chat_openai_compat(messages, system, tools, max_tokens)

    def _chat_anthropic(
        self,
        messages: list[dict],
        system: str,
        tools: list[dict] | None,
        max_tokens: int,
    ) -> LLMResponse:
        """Chat using Anthropic API."""
        kwargs = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": messages,
        }
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = tools

        response = self.client.messages.create(**kwargs)
        
        # Already in correct format
        return LLMResponse(
            content=response.content,
            stop_reason=response.stop_reason,
        )

    def _chat_openai_compat(
        self,
        messages: list[dict],
        system: str,
        tools: list[dict] | None,
        max_tokens: int,
    ) -> LLMResponse:
        """Chat using OpenAI-compatible API (Groq, OpenAI, Ollama)."""
        
        # Convert messages format
        oai_messages = []
        if system:
            oai_messages.append({"role": "system", "content": system})
        
        for msg in messages:
            converted = self._convert_message_to_openai(msg)
            # Handle case where conversion returns a list (tool results)
            if isinstance(converted, list):
                oai_messages.extend(converted)
            else:
                oai_messages.append(converted)
        
        kwargs = {
            "model": self.model,
            "messages": oai_messages,
        }
        
        # o-series models (o1, o3, o4) use max_completion_tokens instead of max_tokens
        is_o_series = self.model.startswith(("o1", "o3", "o4", "gpt-5"))
        if is_o_series:
            kwargs["max_completion_tokens"] = max_tokens
        else:
            kwargs["max_tokens"] = max_tokens
        
        # Convert tools to OpenAI format
        if tools:
            kwargs["tools"] = [self._convert_tool_to_openai(t) for t in tools]
            kwargs["tool_choice"] = "auto"
        
        response = self.client.chat.completions.create(**kwargs)
        
        # Convert response to unified format
        return self._convert_response_from_openai(response)

    def _convert_message_to_openai(self, msg: dict) -> dict | list[dict]:
        """Convert Anthropic message format to OpenAI format.
        
        Returns a single dict or a list of dicts (for tool results).
        """
        role = msg["role"]
        content = msg["content"]
        
        # Handle tool results - OpenAI needs role: "tool" for each result
        if role == "user" and isinstance(content, list):
            tool_results = [c for c in content if c.get("type") == "tool_result"]
            if tool_results:
                # Return list of tool messages
                return [
                    {
                        "role": "tool",
                        "tool_call_id": tr["tool_use_id"],
                        "content": str(tr.get("content", "")),  # Ensure string
                    }
                    for tr in tool_results
                ]
        
        # Handle assistant messages with tool use
        if role == "assistant" and isinstance(content, list):
            text_parts = []
            tool_calls = []
            
            for block in content:
                if block.get("type") == "text":
                    text_parts.append(block["text"])
                elif block.get("type") == "tool_use":
                    tool_calls.append({
                        "id": block["id"],
                        "type": "function",
                        "function": {
                            "name": block["name"],
                            "arguments": json.dumps(block["input"]),
                        }
                    })
            
            # OpenAI requires content to be a string, not null
            result = {"role": "assistant", "content": "\n".join(text_parts) if text_parts else ""}
            if tool_calls:
                result["tool_calls"] = tool_calls
            return result
        
        # Simple text message
        if isinstance(content, str):
            return {"role": role, "content": content}
        elif isinstance(content, list) and len(content) > 0:
            # Extract text from content blocks
            texts = [c.get("text", str(c)) for c in content if isinstance(c, dict)]
            return {"role": role, "content": "\n".join(texts) if texts else "(no content)"}
        
        return {"role": role, "content": str(content) if content else "(no content)"}

    def _convert_tool_to_openai(self, tool: dict) -> dict:
        """Convert Anthropic tool format to OpenAI format."""
        return {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": tool.get("input_schema", {"type": "object", "properties": {}}),
            }
        }

    def _convert_response_from_openai(self, response) -> LLMResponse:
        """Convert OpenAI response to unified format."""
        choice = response.choices[0]
        message = choice.message
        
        content = []
        
        # Add text content
        if message.content:
            content.append(TextBlock(type="text", text=message.content))
        
        # Add tool calls
        if message.tool_calls:
            for tc in message.tool_calls:
                content.append(ToolUseBlock(
                    type="tool_use",
                    id=tc.id,
                    name=tc.function.name,
                    input=json.loads(tc.function.arguments),
                ))
        
        # Map finish reason
        stop_reason = "end_turn"
        if choice.finish_reason == "tool_calls":
            stop_reason = "tool_use"
        elif choice.finish_reason == "length":
            stop_reason = "max_tokens"
        
        return LLMResponse(content=content, stop_reason=stop_reason)
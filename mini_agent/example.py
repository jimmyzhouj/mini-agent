"""
Runnable example entry point.

Usage:
    python -m mini_agent.example "List all Python files and show their line counts"
    python -m mini_agent.example --base-url http://internal:8080/v1 "Fix the bug in app.py"
    python -m mini_agent.example --session-dir ./sessions "Remember what we talked about"
"""

from __future__ import annotations

import argparse
import asyncio
import json

from mini_agent.agent import Agent
from mini_agent.session import SessionManager
from mini_agent.tools.bash import BashTool
from mini_agent.tools.edit import EditTool
from mini_agent.tools.read import ReadTool
from mini_agent.tools.write import WriteTool
from mini_agent.types import AgentEvent, EventType, ProviderConfig


def print_event(event: AgentEvent) -> None:
    """Simple terminal event printer."""
    match event.type:
        case EventType.TEXT_DELTA:
            print(event.data.get("text", ""), end="", flush=True)
        case EventType.TOOL_CALL_START:
            input_str = json.dumps(event.data.get("input", {}), ensure_ascii=False)[:200]
            print(f"\n[tool] {event.data['name']}({input_str})")
        case EventType.TOOL_CALL_END:
            status = "ERROR" if event.data.get("is_error") else "ok"
            print(f"  [{status}] {event.data.get('output', '')[:200]}")
        case EventType.AGENT_START:
            print("[agent start]")
        case EventType.AGENT_END:
            print(f"\n[agent end: {event.data.get('reason', '')}]")
        case EventType.ERROR:
            print(f"\n[error] {event.data.get('error', '')}")
        case EventType.COMPACTION:
            print("\n[context compacted]")


def main() -> None:
    parser = argparse.ArgumentParser(description="Mini Agent")
    parser.add_argument(
        "prompt",
        nargs="?",
        default="What files are in the current directory?",
    )
    parser.add_argument("--model", default="claude-sonnet-4-20250514")
    parser.add_argument("--api-key", default=None, help="API key (default: ANTHROPIC_API_KEY env)")
    parser.add_argument("--base-url", default=None, help="Custom API endpoint URL")
    parser.add_argument("--session-dir", default=None, help="Session directory for persistence")
    parser.add_argument(
        "--tools",
        default="bash,read,write,edit",
        help="Comma-separated tool list",
    )
    args = parser.parse_args()

    config = ProviderConfig(
        api_key=args.api_key,
        base_url=args.base_url,
        model=args.model,
    )

    # Session manager (optional)
    session_mgr: SessionManager | None = None
    if args.session_dir:
        session_mgr = SessionManager.continue_recent(args.session_dir)
        print(f"[session] {session_mgr._path or 'in-memory'}")

    # Select tools
    available = {
        "bash": BashTool(),
        "read": ReadTool(),
        "write": WriteTool(),
        "edit": EditTool(),
    }
    selected_tools = [
        available[t.strip()]
        for t in args.tools.split(",")
        if t.strip() in available
    ]

    # Create agent
    agent = Agent(
        provider_config=config,
        tools=selected_tools,
        session_manager=session_mgr,
    )
    agent.subscribe(print_event)

    # Run
    result = asyncio.run(agent.prompt(args.prompt))

    if result and result.usage:
        u = result.usage
        print(
            f"\n[Tokens: {u.total_tokens} | In: {u.input_tokens} | Out: {u.output_tokens}]"
        )


if __name__ == "__main__":
    main()

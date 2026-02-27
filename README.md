# Mini-Agent

A minimal Python AI Coding Agent framework inspired by the [Pi (badlogic/pi-mono)](https://github.com/badlogic/pi-mono) architecture.
Pure Python implementation, integrating exclusively with Anthropic's Claude models.

---

## Design Philosophy

**Minimal**: Only implement what is needed. No MCP, no plan mode, no plugin system.
The core does one thing: **a while-True loop** — call the LLM, execute tools if there are tool_use blocks, feed results back, repeat; stop when there are no tool_use blocks.

Three core design decisions borrowed from Pi:

1. **output/details separation**: `ToolResult.output` is plain text for the LLM; `ToolResult.details` is structured data for the UI/caller. The LLM does not need to see UI information.
2. **Self-healing parameter validation**: Tool parameters are validated with Pydantic. On failure, the error message is fed back as a `tool_result` so the model can correct itself — no exceptions thrown.
3. **Session as an append-only JSONL tree**: Each record has an `id + parent_id`, supporting branch rollback. A crash loses at most one line.

---

## Directory Structure

```
mini_agent/
├── types.py               # All shared data types (Pydantic)
├── loop.py                # agent_loop() ← core, the only while True
├── agent.py               # Agent class, high-level wrapper
├── session.py             # SessionManager, JSONL persistence
├── compact.py             # compact_messages(), context compaction
├── providers/
│   └── anthropic.py       # AnthropicProvider, the only module that talks to the LLM
└── tools/
    ├── base.py            # Tool base class + ToolRegistry
    ├── bash.py            # bash tool
    ├── read.py            # read tool
    ├── write.py           # write tool
    └── edit.py            # edit tool
tests/
├── conftest.py            # MockProvider, EchoTool, and other shared fixtures
├── test_loop.py           # agent_loop unit tests (mock LLM)
├── test_tools.py          # Execution tests for all four tools
└── test_session.py        # SessionManager persistence tests
```

---

## Quick Start

```bash
pip install -e ".[dev]"
export ANTHROPIC_API_KEY=sk-ant-...

# Simplest usage
python -m mini_agent.example "List Python files and show their line counts"

# With session persistence (resume context in future conversations)
python -m mini_agent.example --session-dir ./sessions "Create hello.py"
python -m mini_agent.example --session-dir ./sessions "Add a docstring to it"

# Custom internal endpoint
python -m mini_agent.example --base-url http://internal:8080/v1 "Fix the bug"
```

**Using in code:**

```python
import asyncio
from mini_agent import Agent, ProviderConfig

agent = Agent(provider_config=ProviderConfig())  # reads ANTHROPIC_API_KEY

# Subscribe to events (optional)
agent.subscribe(lambda e: print(e.type, e.data))

result = asyncio.run(agent.prompt("List all .py files here"))
print(result.usage.total_tokens)
```

---

## Data Flow

Full execution path of a single `agent.prompt("Fix the bug in app.py")` call:

```
agent.prompt("Fix the bug")
  │
  ├─ Append user message to self._messages
  ├─ Persist to SessionManager (if configured)
  │
  └─► agent_loop(provider, messages, tool_registry, ...)
        │
        ├─[loop start]──────────────────────────────────────────────────────┐
        │                                                                    │
        ├─ Check abort_signal                                                │
        ├─ emit(TURN_START)                                                  │
        │                                                                    │
        ├─► _call_llm()                                                      │
        │     └─► provider.stream_response()   ← streaming call             │
        │           ├─ each text delta → emit(TEXT_DELTA)                   │
        │           ├─ tool_use block found → emit(TOOL_CALL_START)         │
        │           └─ returns complete AssistantMessage                     │
        │                                                                    │
        ├─ messages.append(assistant_message)                                │
        │                                                                    │
        ├─[any tool_use?]                                                    │
        │   │                                                                │
        │   ├─ NO → emit(TURN_END)                                           │
        │   │        queued messages? → inject and continue ───────────────►─┘
        │   │        none → emit(AGENT_END) → return last_response
        │   │
        │   └─ YES → execute each tool sequentially:
        │             ├─ tool_registry.get(name)       # look up tool
        │             ├─ tool.validate_params(input)    # Pydantic validation
        │             │   failure → ToolResult(is_error=True, output=error_msg)
        │             │   success → tool.execute(validated_params)
        │             ├─ emit(TOOL_CALL_END)
        │             └─ collect all tool_results
        │
        ├─ messages.append({"role":"user", "content": tool_results})
        ├─ emit(TURN_END)
        └─► back to loop top ───────────────────────────────────────────────┘

  Back in agent.prompt():
  ├─ Persist new messages to SessionManager
  └─ _maybe_compact()  → if over threshold, call compact_messages()
```

**Anthropic API message format requirements** (handled by the provider layer):
- Messages must alternate between user/assistant roles
- `tool_result` is sent with `role: "user"`, not as a separate role
- Every `tool_use` must have a corresponding `tool_result`

---

## Module Reference

### `types.py` — All Shared Types

The project's "vocabulary". All modules share types defined here. All are Pydantic BaseModels and JSON-serializable.

```
ContentBlock = TextBlock | ImageBlock | ToolUseBlock | ToolResultBlock
Message      = UserMessage | AssistantMessage

ToolResult   → output (str, for LLM) + details (dict, for UI)
AgentEvent   → type (EventType enum) + data (dict)
ProviderConfig → api_key, base_url, model, max_tokens, timeout, default_headers
```

**EventType enum:**

| Event | When it fires |
|---|---|
| `AGENT_START` | agent_loop begins |
| `TURN_START/END` | Before/after each LLM call |
| `TEXT_DELTA` | Streaming text chunk (character-level) |
| `TOOL_CALL_START` | Tool begins execution (includes name and input) |
| `TOOL_CALL_END` | Tool finishes execution (includes output summary and is_error) |
| `TOOL_RESULT` | Same as above, includes details (structured, no output text) |
| `ERROR` | LLM call exception |
| `COMPACTION` | Context was compacted |
| `AGENT_END` | Loop ends (reason: completed/aborted/error) |

---

### `loop.py` — Core Loop

**The most important file in the entire framework**, approximately 120 lines.

```python
async def agent_loop(
    provider, system_prompt, messages,   # the three essentials
    tool_registry,                       # available tools
    on_event,                            # event callback (subscribed by UI layer)
    get_queued_messages,                 # injection point for steer/followUp
    abort_signal,                        # asyncio.Event, set() to abort
) -> AssistantMessage
```

**Key behaviors:**
- The `messages` list is **mutated in place** (append-only); callers can hold a reference and observe it directly
- Multiple `tool_use` blocks in the same turn are executed **sequentially**, not in parallel, simplifying state management
- `get_queued_messages()` is called after each turn; if there is content, the loop continues (steer scenario)
- `_call_llm()` prefers streaming; only non-`anthropic.APIError` exceptions fall back to non-streaming (for internal endpoints that do not support SSE)

---

### `providers/anthropic.py` — LLM Interface Layer

**The only module that interacts with the LLM API.** All other code is unaware of API details.

**`stream_response()` implementation strategy**: Anthropic SDK's `messages.stream()` is a synchronous context manager, wrapped via `asyncio.run_in_executor()` into a thread pool to make it asyncio-friendly:

```python
loop = asyncio.get_running_loop()
await loop.run_in_executor(None, _run_stream)  # sync SDK runs in a thread
```

Streaming event handling:
- `content_block_start` (tool_use) → collect `{id, name, input_json=""}`, emit `TOOL_CALL_START`
- `content_block_delta` (text_delta) → accumulate text, emit `TEXT_DELTA`
- `content_block_delta` (input_json_delta) → accumulate tool JSON input
- Stream end → use `stream.get_final_message()` for the complete message (most accurate), parse input_json to dict

**`complete()`** is used for compaction and other scenarios that do not need streaming, also made async via `run_in_executor`.

---

### `tools/base.py` — Tool Base Class

```python
class Tool(ABC, Generic[T]):
    name: str
    description: str
    parameters: Type[T]           # a Pydantic BaseModel class

    def validate_params(raw_input: dict) -> T | ToolResult:
        # success → returns Pydantic object
        # failure → returns ToolResult(is_error=True), error fed back to LLM

    async def execute(params: T) -> ToolResult: ...
```

`validate_params` is Pi's core design: **validation failure does not raise an exception** — instead, Pydantic's error message is sent back as a `tool_result` so the model can correct its parameters and retry.

`ToolRegistry` is a simple `dict[str, Tool]` that looks up tools by name.

---

### Built-in Tools

**bash** (`tools/bash.py`)
- Executes asynchronously via `asyncio.create_subprocess_shell()`, with `wait_for()` for timeout control
- Non-zero exit codes do **not** set `is_error=True` (the LLM decides whether the command succeeded)
- Truncation: keeps the last **2000 lines**, capped at **50KB** (whichever is smaller)

**read** (`tools/read.py`)
- Supports `offset` (1-indexed) / `limit` parameters, defaults to reading the first 2000 lines
- Output includes line numbers: `"1: first line\n2: second line\n..."`
- File header: `"File: {path} ({total_lines} lines)\n"`

**write** (`tools/write.py`)
- `path.parent.mkdir(parents=True, exist_ok=True)` creates directories automatically
- `details["created"]` tells the caller whether the file was newly created or overwritten

**edit** (`tools/edit.py`)
- `old_text` must appear in the file **exactly once**; otherwise an error is returned
- Forces the LLM to `read` before `edit`, ensuring precision
- 0 matches → `"old_text not found"`; >1 match → `"found N times, must be unique"`

---

### `agent.py` — High-Level Wrapper

`Agent` exposes a clean API and manages all internal state:

```python
agent = Agent(provider_config=config, tools=[...], session_manager=sm)
unsubscribe = agent.subscribe(my_callback)

# Send a message
result = await agent.prompt("Fix the bug")

# In-flight control (can be called from other coroutines/threads)
agent.steer("Actually, focus on test.py instead")   # inject immediately, clear followup queue
agent.follow_up("Run the tests after you're done")  # inject after natural completion
agent.abort()                                        # abort the current loop
```

**`steer` vs `follow_up` priority:**
- `steer` injects after the current tool finishes, before the next LLM call, and **clears the followup queue** (user actively intervening)
- `follow_up` only injects after the agent finishes naturally, before the next `prompt()` call

**Compaction trigger:** `_maybe_compact()` is called after `prompt()` completes. It estimates token count (`characters / 4`) and triggers compaction if the 100k token threshold is exceeded.

---

### `session.py` — JSONL Tree Persistence

One JSON object per line; each record has an `id` and `parent_id`, forming a tree:

```jsonl
{"id":"a1b2c3","parent_id":null,"type":"user","data":{"content":"Fix the bug"},...}
{"id":"d4e5f6","parent_id":"a1b2c3","type":"assistant","data":{"content":[...]},...}
{"id":"g7h8i9","parent_id":"d4e5f6","type":"tool_result","data":{"results":[...]},...}
```

**Core concepts:**
- `leaf_id`: the ID of the current branch's leaf node
- `build_context()`: traces from leaf up to root, reverses, and builds the messages list
- `branch(entry_id)`: moves the leaf to a specified node; subsequent appends fork from there (no data is deleted)
- Crash recovery: `_load()` uses try/except to skip corrupted lines — at most one line of data is lost, no crash

**Factory methods:**
```python
SessionManager.in_memory()              # discarded on process exit
SessionManager.create("./sessions")     # creates a new session file (timestamp-named)
SessionManager.open("./s/session.jsonl")# opens an existing file
SessionManager.continue_recent("./s")  # opens the most recently modified, or creates one
```

---

### `compact.py` — Context Compaction

When the message list grows too long, the LLM generates a summary to replace older messages:

```
Original messages: [old1, old2, old3, ..., recent-4, recent-3, recent-2, recent-1]
                   |_______ sent to LLM for summary ________|  |____ kept as-is ___|

After compaction: [{"role":"user","content":"[Previous conversation summary]\n{summary}"},
                   recent-4, recent-3, recent-2, recent-1]
```

The most recent 4 messages are kept (`keep_recent=4`) to ensure the latest context is intact.
The compacted list always starts with a user message (required by the Anthropic API).

---

## Testing

```bash
pytest tests/ -v
```

| File | What it tests |
|---|---|
| `test_loop.py` | 8 scenarios using `MockProvider` to simulate the LLM — no real API calls |
| `test_tools.py` | Real filesystem (`tmp_path`); bash uses `python -c` for cross-platform compatibility |
| `test_session.py` | JSONL persistence, branch, crash recovery |

**How `MockProvider` works:**
```python
# Pre-configure a sequence of LLM responses
provider = MockProvider([
    AssistantMessage(content=[ToolUseBlock(...)]),  # 1st call
    AssistantMessage(content=[TextBlock("Done")]),  # 2nd call
])
# Each stream_response() pops one response from the queue
```

---

## Extension: Custom Tools

```python
from pydantic import BaseModel, Field
from mini_agent.tools.base import Tool
from mini_agent.types import ToolResult

class SearchParams(BaseModel):
    query: str = Field(description="Search query")
    limit: int = Field(default=10, description="Max results")

class SearchTool(Tool[SearchParams]):
    name = "search"
    description = "Search the codebase for a pattern"
    parameters = SearchParams

    async def execute(self, params: SearchParams) -> ToolResult:
        results = do_search(params.query, params.limit)
        return ToolResult(
            output="\n".join(results),          # what the LLM sees
            details={"count": len(results)},    # what the UI sees
        )

agent = Agent(
    provider_config=ProviderConfig(),
    tools=[BashTool(), ReadTool(), SearchTool()],
)
```

---

## Explicitly Out of Scope

- Multi-provider support (Anthropic only)
- MCP / plugin system
- TUI interface
- Thinking / reasoning trace
- Token cost calculation
- Parallel tool execution
- Max-steps limit

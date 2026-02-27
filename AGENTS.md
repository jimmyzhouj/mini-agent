# Mini-Agent：Python 实现的最小 AI Coding Agent 框架

## 项目概述

### 目标

构建一个基于 Pi (badlogic/pi-mono) 架构设计的最小 Python Agent 框架。核心理念是照搬 Pi 的设计决策（agent loop、tool 机制、session 管理），但用纯 Python 实现，只对接 Anthropic 格式的 Claude 模型（支持指向内部 endpoint）。

### 设计原则（来自 Pi）

1. **极简**：不需要的功能不做。没有 max-steps 限制、没有 plan mode、没有 MCP。
2. **agent loop 就是 while True**：调用 LLM → 有 tool_use 就执行 → 回灌结果 → 继续；没有 tool_use 就结束。
3. **Tool output/details 分离**：给 LLM 的是纯文本 `output`，给 UI/调用方的是结构化 `details`。LLM 不需要看 UI 信息。
4. **顺序执行 tool call**：同一轮的多个 tool call 按顺序逐一执行，不并行。简化状态管理。
5. **参数验证自修复**：tool 参数用 Pydantic 验证，验证失败的错误信息作为 tool_result 回灌给 LLM，让模型自行修正。
6. **Session 是 append-only JSONL 树**：每条记录有 id + parent_id，支持 branch 回退，crash-safe。
7. **Compaction**：context 接近上限时自动摘要压缩，保留最近几轮原始内容。

### 技术栈

- Python 3.11+
- `anthropic` SDK（官方 Python SDK）
- `pydantic` v2（参数验证 + 数据模型）
- 标准库：`asyncio`, `subprocess`, `json`, `uuid`, `pathlib`, `logging`
- 无其他第三方依赖

---

## 项目结构

```
mini-agent/
├── pyproject.toml
├── README.md
├── AGENTS.md                    # 本文件
├── mini_agent/
│   ├── __init__.py              # 版本号 + 顶层导出
│   ├── types.py                 # 所有核心类型定义
│   ├── loop.py                  # agentLoop() — 核心 while True 异步生成器
│   ├── agent.py                 # Agent 类 — state管理 + prompt() + steer/followUp
│   ├── session.py               # SessionManager — JSONL 树状会话持久化
│   ├── compact.py               # Compaction 逻辑 — context 压缩
│   ├── providers/
│   │   ├── __init__.py
│   │   └── anthropic.py         # Anthropic API 封装（支持自定义 base_url）
│   ├── tools/
│   │   ├── __init__.py          # 导出所有内置 tool
│   │   ├── base.py              # Tool 基类 + 注册机制
│   │   ├── bash.py              # bash 工具
│   │   ├── read.py              # read 工具
│   │   ├── write.py             # write 工具
│   │   └── edit.py              # edit 工具
│   └── example.py               # 可直接运行的示例入口
└── tests/
    ├── test_loop.py             # agent loop 测试（mock LLM）
    ├── test_tools.py            # tool 执行测试
    ├── test_session.py          # session 持久化测试
    └── conftest.py              # pytest fixtures
```

---

## 一、核心类型定义 (`types.py`)

本文件定义所有共享的数据结构。使用 Pydantic BaseModel，所有类型可 JSON 序列化。

### 消息类型

```python
from pydantic import BaseModel, Field
from typing import Literal, Any
from enum import Enum
import time
import uuid

# --- Content Block 类型 ---

class TextBlock(BaseModel):
    type: Literal["text"] = "text"
    text: str

class ImageBlock(BaseModel):
    type: Literal["image"] = "image"
    data: str          # base64 编码
    mime_type: str     # "image/png", "image/jpeg" 等

class ToolUseBlock(BaseModel):
    type: Literal["tool_use"] = "tool_use"
    id: str            # tool call ID，由 LLM 生成
    name: str          # tool 名称
    input: dict        # tool 参数（原始 dict，未验证）

class ToolResultBlock(BaseModel):
    type: Literal["tool_result"] = "tool_result"
    tool_use_id: str
    content: str       # 给 LLM 看的纯文本
    is_error: bool = False

ContentBlock = TextBlock | ImageBlock | ToolUseBlock | ToolResultBlock

# --- Message 类型 ---

class UserMessage(BaseModel):
    role: Literal["user"] = "user"
    content: str | list[ContentBlock]
    timestamp: float = Field(default_factory=time.time)

class AssistantMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: list[ContentBlock]
    stop_reason: str | None = None          # "end_turn" | "tool_use" | "max_tokens"
    model: str | None = None
    usage: "TokenUsage | None" = None
    timestamp: float = Field(default_factory=time.time)

class TokenUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

Message = UserMessage | AssistantMessage
```

### Tool 执行结果

```python
class ToolResult(BaseModel):
    """Tool 执行的返回值。output 给 LLM，details 给调用方/UI。"""
    output: str                  # 纯文本，会作为 tool_result content 发给 LLM
    details: dict[str, Any] = {} # 结构化数据，不发给 LLM，供 UI 或日志使用
    is_error: bool = False
```

### Agent 事件

```python
class EventType(str, Enum):
    AGENT_START = "agent_start"
    AGENT_END = "agent_end"
    TURN_START = "turn_start"
    TURN_END = "turn_end"
    TEXT_DELTA = "text_delta"           # LLM 流式文本片段
    TOOL_CALL_START = "tool_call_start"
    TOOL_CALL_END = "tool_call_end"
    TOOL_RESULT = "tool_result"
    ERROR = "error"
    COMPACTION = "compaction"           # context 被压缩时触发

class AgentEvent(BaseModel):
    type: EventType
    data: dict[str, Any] = {}
    timestamp: float = Field(default_factory=time.time)
```

### Agent 状态

```python
class AgentState(BaseModel):
    system_prompt: str
    model: str                          # model ID, 如 "claude-sonnet-4-20250514"
    messages: list[dict] = []           # Anthropic SDK 格式的消息列表
    tools: list[str] = []              # 已注册 tool 的名称列表
    total_usage: TokenUsage = Field(default_factory=TokenUsage)
```

### Provider 配置

```python
class ProviderConfig(BaseModel):
    """Anthropic provider 配置。支持指向内部 endpoint。"""
    api_key: str | None = None          # None 则从环境变量读取
    base_url: str | None = None         # None 则使用官方 endpoint；填内部 URL 则指向内部服务
    model: str = "claude-sonnet-4-20250514"
    max_tokens: int = 8096
    timeout: float = 300.0              # 请求超时（秒）
    default_headers: dict[str, str] = {}  # 内部 endpoint 可能需要的额外 header
```

---

## 二、Anthropic Provider (`providers/anthropic.py`)

封装 Anthropic SDK 调用。**这是唯一与 LLM API 交互的模块**，其余代码通过此模块间接使用 LLM。

### 职责

1. 初始化 `anthropic.Anthropic` 客户端（支持自定义 `base_url` 和 `api_key`）
2. 将内部 Tool 定义转换为 Anthropic SDK 格式
3. 提供 `stream_response()` 方法：流式调用，yield 标准化事件
4. 提供 `complete()` 方法：非流式调用，返回完整 AssistantMessage
5. 提供 `summarize()` 方法：用于 compaction，发送摘要请求

### 关键接口

```python
class AnthropicProvider:
    def __init__(self, config: ProviderConfig):
        """
        初始化。内部创建 anthropic.Anthropic(
            api_key=config.api_key,      # None 则 SDK 自动读 ANTHROPIC_API_KEY
            base_url=config.base_url,    # None 则用官方 endpoint
            timeout=config.timeout,
            default_headers=config.default_headers,
        )
        """

    def _build_tools(self, tools: list["Tool"]) -> list[dict]:
        """
        将 Tool 对象列表转为 Anthropic SDK 格式：
        [{ "name": "bash", "description": "...", "input_schema": { Pydantic model 的 JSON Schema } }]

        关键：用 tool.parameters.model_json_schema() 获取 JSON Schema。
        """

    async def stream_response(
        self,
        messages: list[dict],
        system_prompt: str,
        tools: list["Tool"],
        model: str | None = None,
    ) -> AsyncGenerator[AgentEvent, None]:
        """
        流式调用 LLM。使用 client.messages.stream() 上下文管理器。

        事件映射：
        - InputJSON delta → 不直接 yield，累积到当前 tool_call
        - text delta → yield TEXT_DELTA 事件
        - content_block_start (type=tool_use) → yield TOOL_CALL_START
        - message_stop → yield 最终的 AssistantMessage

        返回：AsyncGenerator[AgentEvent, None]
        最终（最后一个 yield）产出一个 data 中包含完整 AssistantMessage 的事件。

        异常处理：
        - anthropic.APIError → yield ERROR 事件
        - asyncio.CancelledError → 正常退出（abort 支持）
        """

    async def complete(
        self,
        messages: list[dict],
        system_prompt: str,
        tools: list["Tool"] | None = None,
        model: str | None = None,
    ) -> AssistantMessage:
        """
        非流式调用。用于 compaction 等不需要流式的场景。
        使用 client.messages.create()。
        """
```

### 消息格式转换

Anthropic SDK 接受的 messages 格式：

```python
# user message
{"role": "user", "content": "Fix the bug"}
# 或带图片
{"role": "user", "content": [
    {"type": "text", "text": "What's in this image?"},
    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "..."}}
]}

# assistant message（包含 tool_use）
{"role": "assistant", "content": [
    {"type": "text", "text": "I'll read the file first."},
    {"type": "tool_use", "id": "toolu_xxx", "name": "read", "input": {"path": "app.py"}}
]}

# tool result（必须紧跟在含 tool_use 的 assistant message 之后）
{"role": "user", "content": [
    {"type": "tool_result", "tool_use_id": "toolu_xxx", "content": "file contents..."}
]}
```

**注意**：Anthropic 的 tool_result 是作为 `role: "user"` 消息发送的，content 是 `tool_result` block 数组。这与 OpenAI 的 `role: "tool"` 不同。Provider 层负责屏蔽这个细节。

---

## 三、Tool 系统 (`tools/base.py` + 各工具文件)

### Tool 基类

```python
from pydantic import BaseModel
from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Type, Any

T = TypeVar("T", bound=BaseModel)

class Tool(ABC, Generic[T]):
    """
    所有 tool 的基类。仿照 Pi 的 AgentTool 接口。

    子类需要实现：
    - name: str
    - description: str
    - parameters: Type[T]  (一个 Pydantic BaseModel 类)
    - execute(params: T) -> ToolResult
    """
    name: str
    description: str
    parameters: Type[T]

    def get_schema(self) -> dict:
        """返回 Anthropic 格式的 tool 定义。"""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.parameters.model_json_schema(),
        }

    def validate_params(self, raw_input: dict) -> T | ToolResult:
        """
        用 Pydantic 验证参数。
        成功 → 返回验证后的 Pydantic 对象。
        失败 → 返回 ToolResult(is_error=True)，包含详细错误信息。

        这是 Pi 的核心设计：验证失败不抛异常，而是把错误信息
        作为 tool_result 回灌给 LLM，让模型自行修正参数。
        """
        try:
            return self.parameters.model_validate(raw_input)
        except ValidationError as e:
            return ToolResult(
                output=f"Parameter validation failed:\n{e}",
                is_error=True,
            )

    @abstractmethod
    async def execute(self, params: T) -> ToolResult:
        """执行 tool。子类实现。"""
        ...
```

### ToolRegistry

```python
class ToolRegistry:
    """Tool 注册表。管理所有可用 tools。"""

    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def list_tools(self) -> list[Tool]:
        return list(self._tools.values())

    def get_schemas(self) -> list[dict]:
        """返回所有 tool 的 Anthropic 格式 schema 列表。"""
        return [t.get_schema() for t in self._tools.values()]
```

### bash 工具 (`tools/bash.py`)

```python
class BashParams(BaseModel):
    command: str = Field(description="Bash command to execute")
    timeout: int | None = Field(default=None, description="Timeout in seconds")

class BashTool(Tool[BashParams]):
    name = "bash"
    description = "Execute a bash command in the working directory. Returns stdout and stderr."
    parameters = BashParams

    def __init__(self, working_dir: str | None = None):
        self.working_dir = working_dir or os.getcwd()

    async def execute(self, params: BashParams) -> ToolResult:
        """
        实现要点：
        1. 使用 asyncio.create_subprocess_shell() 异步执行
        2. 捕获 stdout + stderr
        3. 超时处理：params.timeout 秒后 kill 进程
        4. 输出截断：只保留最后 2000 行 或 50KB（取小的）
           截断时头部插入 "[truncated: showing last 2000 of N lines]"
        5. 返回 ToolResult:
           - output: 格式化的文本（给 LLM 看）
             成功: "stdout:\n{stdout}\nstderr:\n{stderr}"
             失败: "Exit code: {code}\nstdout:\n{stdout}\nstderr:\n{stderr}"
           - details: {"exit_code": int, "stdout": str, "stderr": str, "truncated": bool}
           - is_error: False（即使 exit code != 0 也不标记 error，让 LLM 自行判断）
             只在进程启动失败等系统级错误时标记 is_error=True
        """
```

### read 工具 (`tools/read.py`)

```python
class ReadParams(BaseModel):
    path: str = Field(description="Path to the file to read (relative or absolute)")
    offset: int | None = Field(default=None, description="Line number to start reading from (1-indexed)")
    limit: int | None = Field(default=None, description="Maximum number of lines to read")

class ReadTool(Tool[ReadParams]):
    name = "read"
    description = (
        "Read the contents of a file. For text files, defaults to first 2000 lines. "
        "Use offset/limit for large files. Supports images (jpg, png, gif, webp) "
        "which are returned as base64."
    )
    parameters = ReadParams

    def __init__(self, working_dir: str | None = None):
        self.working_dir = working_dir or os.getcwd()

    async def execute(self, params: ReadParams) -> ToolResult:
        """
        实现要点：
        1. 解析路径：相对路径基于 working_dir 解析
        2. 检查文件是否存在
        3. 判断是否图片（通过后缀 .jpg/.jpeg/.png/.gif/.webp）
           - 图片：读取为 base64，返回特殊格式（details 中标记 type="image"）
           - 文本：读取内容，应用 offset/limit
        4. 默认 limit=2000 行
        5. 截断到 50KB 上限
        6. 输出格式：带行号
           "File: {path} ({total_lines} lines)\n1: first line\n2: second line\n..."
        7. details: {"path": str, "total_lines": int, "shown_lines": int, "truncated": bool}
        """
```

### write 工具 (`tools/write.py`)

```python
class WriteParams(BaseModel):
    path: str = Field(description="Path to the file to write (relative or absolute)")
    content: str = Field(description="Content to write to the file")

class WriteTool(Tool[WriteParams]):
    name = "write"
    description = "Write content to a file. Creates the file if it doesn't exist, overwrites if it does. Automatically creates parent directories."
    parameters = WriteParams

    async def execute(self, params: WriteParams) -> ToolResult:
        """
        实现要点：
        1. 解析路径
        2. 自动创建父目录：Path(path).parent.mkdir(parents=True, exist_ok=True)
        3. 写入文件
        4. output: "Wrote {n} bytes to {path}"
        5. details: {"path": str, "bytes_written": int, "created": bool}
        """
```

### edit 工具 (`tools/edit.py`)

```python
class EditParams(BaseModel):
    path: str = Field(description="Path to the file to edit (relative or absolute)")
    old_text: str = Field(description="Exact text to find and replace (must match exactly)")
    new_text: str = Field(description="New text to replace the old text with")

class EditTool(Tool[EditParams]):
    name = "edit"
    description = (
        "Edit a file by replacing exact text. The old_text must match exactly "
        "(including whitespace). Use this for precise, surgical edits."
    )
    parameters = EditParams

    async def execute(self, params: EditParams) -> ToolResult:
        """
        实现要点：
        1. 读取文件全部内容
        2. 检查 old_text 在文件中出现的次数：
           - 0 次 → 返回 ToolResult(output="old_text not found in {path}", is_error=True)
           - >1 次 → 返回 ToolResult(output="old_text found {n} times, must be unique", is_error=True)
           - 恰好 1 次 → 执行替换
        3. 写回文件
        4. 生成简易 diff 用于 output：显示替换前后各几行上下文
        5. output: "Edited {path}:\n- {old_text 摘要}\n+ {new_text 摘要}"
        6. details: {"path": str, "old_text": str, "new_text": str}

        Pi 的关键设计：old_text 必须精确匹配，没有 regex、没有 fuzzy matching。
        这迫使 LLM 先 read 文件再 edit，确保精确性。
        """
```

---

## 四、Agent Loop (`loop.py`)

**这是整个框架最核心的文件**，实现 Pi 的 `agentLoop()` 模式。

### 接口设计

```python
async def agent_loop(
    provider: AnthropicProvider,
    system_prompt: str,
    messages: list[dict],
    tool_registry: ToolRegistry,
    model: str | None = None,
    on_event: Callable[[AgentEvent], None] | None = None,
    get_queued_messages: Callable[[], list[dict]] | None = None,
    abort_signal: asyncio.Event | None = None,
) -> AssistantMessage:
    """
    核心 agent 循环。

    参数：
    - provider: Anthropic provider 实例
    - system_prompt: 系统提示词
    - messages: 当前对话上下文（会被原地修改，追加新消息）
    - tool_registry: 已注册的 tool
    - model: 模型 ID（可选，覆盖 provider 默认值）
    - on_event: 事件回调（每个事件触发一次）
    - get_queued_messages: 每轮结束后调用，获取外部注入的消息（steer/followUp）
    - abort_signal: 取消信号，set() 后中止循环

    返回：最后一次 LLM 响应的 AssistantMessage
    """
```

### 实现伪代码（必须严格遵循）

```python
async def agent_loop(...) -> AssistantMessage:
    emit(AGENT_START)
    last_response = None

    while True:
        # 检查 abort
        if abort_signal and abort_signal.is_set():
            emit(AGENT_END, {"reason": "aborted"})
            break

        emit(TURN_START)

        # ① 调用 LLM（流式）
        try:
            assistant_message = await _stream_and_collect(
                provider, messages, system_prompt, tool_registry, model, on_event
            )
        except Exception as e:
            emit(ERROR, {"error": str(e)})
            emit(AGENT_END, {"reason": "error"})
            raise

        last_response = assistant_message

        # ② 追加 assistant message 到 context
        messages.append(_to_api_format(assistant_message))

        # ③ 提取 tool_use blocks
        tool_uses = [b for b in assistant_message.content if isinstance(b, ToolUseBlock)]

        # ④ 无 tool call → 结束
        if not tool_uses:
            emit(TURN_END)
            # 检查是否有排队消息
            if get_queued_messages:
                queued = get_queued_messages()
                if queued:
                    for msg in queued:
                        messages.append(msg)
                    continue  # 有排队消息则继续循环
            emit(AGENT_END, {"reason": "completed"})
            break

        # ⑤ 顺序执行每个 tool call
        tool_results = []
        for tool_use in tool_uses:
            # 检查 abort（每个 tool 之间检查一次）
            if abort_signal and abort_signal.is_set():
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_use.id,
                    "content": "Aborted by user",
                    "is_error": True,
                })
                continue

            emit(TOOL_CALL_START, {"name": tool_use.name, "input": tool_use.input})

            tool = tool_registry.get(tool_use.name)
            if not tool:
                result = ToolResult(output=f"Unknown tool: {tool_use.name}", is_error=True)
            else:
                # 参数验证
                validated = tool.validate_params(tool_use.input)
                if isinstance(validated, ToolResult):
                    # 验证失败，validated 就是错误 ToolResult
                    result = validated
                else:
                    # 验证成功，执行 tool
                    try:
                        result = await tool.execute(validated)
                    except Exception as e:
                        result = ToolResult(output=f"Tool execution error: {e}", is_error=True)

            emit(TOOL_CALL_END, {
                "name": tool_use.name,
                "output": result.output[:500],  # 事件中只放摘要
                "is_error": result.is_error,
            })
            emit(TOOL_RESULT, {"name": tool_use.name, "details": result.details})

            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tool_use.id,
                "content": result.output,
                "is_error": result.is_error,
            })

        # ⑥ 回灌 tool results 到 context
        messages.append({"role": "user", "content": tool_results})

        emit(TURN_END)

        # ⑦ 检查排队消息（steer 场景）
        if get_queued_messages:
            queued = get_queued_messages()
            if queued:
                for msg in queued:
                    messages.append(msg)

        # ⑧ 回到 while True 顶部 → 再次调用 LLM

    return last_response
```

### `_stream_and_collect` 辅助函数

```python
async def _stream_and_collect(
    provider, messages, system_prompt, tool_registry, model, on_event
) -> AssistantMessage:
    """
    调用 provider.stream_response()，收集所有事件，
    组装最终的 AssistantMessage。

    流式过程中通过 on_event 回调转发 TEXT_DELTA 事件。
    """
```

### `_to_api_format` 辅助函数

```python
def _to_api_format(assistant_message: AssistantMessage) -> dict:
    """
    将内部 AssistantMessage 转为 Anthropic API 格式的 dict。
    主要是把 content list 中的 Pydantic 对象转为 plain dict。
    """
```

---

## 五、Agent 类 (`agent.py`)

高层封装，对外暴露简洁的 API。仿照 Pi 的 `Agent` 类。

### 接口

```python
class Agent:
    def __init__(
        self,
        provider_config: ProviderConfig,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        tools: list[Tool] | None = None,
        session_manager: "SessionManager | None" = None,
    ):
        """
        初始化 Agent。

        - 创建 AnthropicProvider
        - 创建 ToolRegistry，注册传入的 tools
        - 如果 tools 为 None，注册默认的 4 个 coding tools
        - 初始化消息列表
        - 初始化 steer_queue 和 followup_queue
        - 绑定 session_manager（可选）
        """

    # --- 核心方法 ---

    async def prompt(self, message: str) -> AssistantMessage:
        """
        发送用户消息并运行 agent loop 直到完成。

        1. 构建 user message，append 到 messages
        2. 如果有 session_manager，记录 user message
        3. 调用 agent_loop()，传入 get_queued_messages=self._drain_queues
        4. 如果有 session_manager，记录 assistant message + tool results
        5. 检查是否需要 compaction
        6. 返回最终 AssistantMessage
        """

    def steer(self, message: str) -> None:
        """
        中断当前执行，注入消息。
        消息在当前 tool 执行完后、下一次 LLM 调用前被注入。
        """
        self._steer_queue.append({"role": "user", "content": message})

    def follow_up(self, message: str) -> None:
        """
        排队消息，在 agent 自然结束后注入。
        不会中断当前执行。
        """
        self._followup_queue.append({"role": "user", "content": message})

    def abort(self) -> None:
        """中止当前 agent loop。"""
        self._abort_signal.set()

    # --- State 管理 ---

    def set_model(self, model: str) -> None:
        """切换模型。下次 prompt 生效。"""

    def set_system_prompt(self, prompt: str) -> None:
        """更新系统提示词。"""

    def set_tools(self, tools: list[Tool]) -> None:
        """替换 tool 集合。"""

    def get_messages(self) -> list[dict]:
        """获取当前消息列表（只读副本）。"""

    def replace_messages(self, messages: list[dict]) -> None:
        """替换消息列表（用于 compaction 后）。"""

    # --- 事件订阅 ---

    def subscribe(self, callback: Callable[[AgentEvent], None]) -> Callable:
        """
        订阅事件。返回取消订阅的函数。
        """

    # --- 私有方法 ---

    def _drain_queues(self) -> list[dict]:
        """
        获取队列中的消息。优先级：steer > followup。
        steer 消息会清空 followup（因为用户主动介入了）。
        followup 消息只在没有 steer 时返回。
        取出后清空对应队列。
        """

    def _emit(self, event: AgentEvent) -> None:
        """向所有订阅者发射事件。"""

    async def _maybe_compact(self) -> None:
        """
        检查 context 是否需要压缩。
        估算当前 messages 的 token 数（粗略：字符数 / 4）。
        如果超过 model context window 的 80%（默认阈值 160000 字符），触发 compaction。
        """
```

### 默认系统提示词

```python
DEFAULT_SYSTEM_PROMPT = """You are an expert coding assistant. You help users with coding tasks by reading files, executing commands, editing code, and writing new files.

Available tools:
- read: Read file contents
- bash: Execute bash commands
- edit: Make surgical edits to files (old text must match exactly)
- write: Create or overwrite files

Guidelines:
- Use bash for file operations like ls, grep, find
- Read files before editing to understand context
- Use edit for precise changes, write only for new files or complete rewrites
- Be concise in your responses
"""
```

---

## 六、Session 管理 (`session.py`)

### JSONL 格式

每行一个 JSON 对象：

```json
{"id": "a1b2c3", "parent_id": null, "type": "user", "data": {"content": "Fix the bug"}, "timestamp": 1234567890.123, "model": null}
{"id": "d4e5f6", "parent_id": "a1b2c3", "type": "assistant", "data": {"content": [...]}, "timestamp": 1234567890.456, "model": "claude-sonnet-4-20250514"}
{"id": "g7h8i9", "parent_id": "d4e5f6", "type": "tool_result", "data": {"tool_use_id": "toolu_xxx", "content": "..."}, "timestamp": 1234567890.789, "model": null}
```

### SessionEntry

```python
class SessionEntry(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    parent_id: str | None = None
    type: Literal["user", "assistant", "tool_result", "meta"]
    data: dict
    timestamp: float = Field(default_factory=time.time)
    model: str | None = None
```

### SessionManager

```python
class SessionManager:
    """
    JSONL 树状会话管理。仿照 Pi 的 SessionManager。

    关键概念：
    - 每个 entry 有 id 和 parent_id，构成一棵树
    - 当前活跃分支由 leaf_id 确定（从 leaf 向上追溯到 root 就是当前对话）
    - branch(entry_id) 把 leaf 移到指定节点，后续 append 从此处分叉
    - 文件是 append-only 的，crash 最多丢一行
    """

    def __init__(self, path: str | None = None):
        """
        path=None → 内存模式（不持久化）
        path=有值 → 读取已有 JSONL 文件 / 创建新文件
        """

    # --- 工厂方法 ---

    @staticmethod
    def in_memory() -> "SessionManager":
        """创建内存会话（进程退出即丢）。"""

    @staticmethod
    def create(session_dir: str) -> "SessionManager":
        """在 session_dir 下创建新的 JSONL 文件（文件名用时间戳）。"""

    @staticmethod
    def open(path: str) -> "SessionManager":
        """打开已有会话文件。"""

    @staticmethod
    def continue_recent(session_dir: str) -> "SessionManager":
        """打开 session_dir 下最近修改的 JSONL 文件。没有则创建新的。"""

    @staticmethod
    def list_sessions(session_dir: str) -> list[dict]:
        """列出目录下所有会话文件，返回 [{path, modified_time, entry_count}]。"""

    # --- 核心方法 ---

    def append(self, entry_type: str, data: dict, model: str | None = None) -> SessionEntry:
        """
        追加一条 entry。
        - 自动生成 id
        - parent_id 设为当前 leaf_id
        - 写入 JSONL 文件（如果有 path）
        - 更新 leaf_id 为新 entry 的 id
        - 返回新 entry
        """

    def branch(self, entry_id: str) -> None:
        """
        分叉。将 leaf_id 设为 entry_id。
        后续 append 会以此 entry 为父节点。
        不删除任何数据（append-only）。
        """

    def build_context(self) -> list[dict]:
        """
        从当前 leaf 向上追溯到 root，构建有序的 messages 列表。
        返回 Anthropic API 格式的 messages。

        算法：
        1. 从 leaf_id 开始，沿 parent_id 链向上走到 root
        2. 反转得到从 root 到 leaf 的有序列表
        3. 将 entry 转为 Anthropic message 格式
        4. 跳过 type="meta" 的 entry（compaction 摘要直接嵌在 context 里）
        """

    def get_leaf(self) -> SessionEntry | None:
        """获取当前分支的叶子节点。"""

    def get_tree(self) -> dict:
        """
        获取完整树结构。
        返回 {id, type, children: [...], data_preview} 的嵌套结构。
        """

    def get_entries(self) -> list[SessionEntry]:
        """获取所有 entry 的平坦列表。"""
```

---

## 七、Compaction (`compact.py`)

### 职责

当 context 过长时，用 LLM 生成摘要替换旧消息。

### 接口

```python
async def compact_messages(
    provider: AnthropicProvider,
    messages: list[dict],
    system_prompt: str,
    keep_recent: int = 4,
    model: str | None = None,
) -> list[dict]:
    """
    压缩消息列表。

    策略（仿照 Pi）：
    1. 保留最近 keep_recent 条消息不动（这些是最新的上下文，不能丢）
    2. 将更早的消息发给 LLM，请求生成摘要
    3. 返回新的消息列表：[摘要 user message] + [保留的最近消息]

    摘要 prompt：
    "Summarize the following conversation history concisely.
     Preserve: key decisions, file paths mentioned, code changes made, errors encountered.
     Drop: verbose tool outputs, intermediate reasoning."

    返回的摘要作为一条 user message 插入：
    {"role": "user", "content": "[Previous conversation summary]\n{summary}"}

    注意：返回的 messages 列表必须以 user message 开头（Anthropic 要求）。
    """
```

### 触发条件

在 `Agent.prompt()` 结束后调用 `_maybe_compact()`：

```python
async def _maybe_compact(self):
    estimated_tokens = sum(len(str(m)) for m in self.messages) // 4
    threshold = 100_000  # 约 100K tokens，可配置

    if estimated_tokens > threshold:
        self._emit(AgentEvent(type=EventType.COMPACTION, data={"estimated_tokens": estimated_tokens}))
        self.messages = await compact_messages(
            self.provider, self.messages, self.system_prompt
        )
        if self.session_manager:
            self.session_manager.append("meta", {
                "type": "compaction",
                "message_count_before": estimated_tokens,
            })
```

---

## 八、示例入口 (`example.py`)

```python
"""
可直接运行的示例：

    python -m mini_agent.example "List all Python files and show their line counts"
    python -m mini_agent.example --base-url http://internal:8080/v1 "Fix the bug in app.py"
"""

import argparse
import asyncio

def main():
    parser = argparse.ArgumentParser(description="Mini Agent")
    parser.add_argument("prompt", nargs="?", default="What files are in the current directory?")
    parser.add_argument("--model", default="claude-sonnet-4-20250514")
    parser.add_argument("--api-key", default=None, help="API key (default: ANTHROPIC_API_KEY env)")
    parser.add_argument("--base-url", default=None, help="Custom API endpoint URL")
    parser.add_argument("--session-dir", default=None, help="Session directory for persistence")
    parser.add_argument("--tools", default="bash,read,write,edit",
                       help="Comma-separated tool list")
    args = parser.parse_args()

    # 构建 provider config
    config = ProviderConfig(
        api_key=args.api_key,
        base_url=args.base_url,
        model=args.model,
    )

    # 构建 session manager
    session_mgr = None
    if args.session_dir:
        session_mgr = SessionManager.continue_recent(args.session_dir)

    # 选择 tools
    available = {"bash": BashTool(), "read": ReadTool(), "write": WriteTool(), "edit": EditTool()}
    selected = [available[t.strip()] for t in args.tools.split(",") if t.strip() in available]

    # 创建 agent
    agent = Agent(
        provider_config=config,
        tools=selected,
        session_manager=session_mgr,
    )

    # 订阅事件 → 打印到终端
    agent.subscribe(print_event)

    # 运行
    result = asyncio.run(agent.prompt(args.prompt))

    # 打印 token 统计
    if result and result.usage:
        print(f"\n[Tokens: {result.usage.total_tokens} | In: {result.usage.input_tokens} | Out: {result.usage.output_tokens}]")

def print_event(event: AgentEvent):
    """简单的终端事件打印器。"""
    match event.type:
        case EventType.TEXT_DELTA:
            print(event.data.get("text", ""), end="", flush=True)
        case EventType.TOOL_CALL_START:
            print(f"\n🔧 {event.data['name']}({json.dumps(event.data.get('input', {}), ensure_ascii=False)[:200]})")
        case EventType.TOOL_CALL_END:
            status = "❌" if event.data.get("is_error") else "✅"
            print(f"   {status} {event.data.get('output', '')[:200]}")
        case EventType.ERROR:
            print(f"\n❗ Error: {event.data.get('error', '')}")
        case EventType.COMPACTION:
            print(f"\n📦 Context compacted")

if __name__ == "__main__":
    main()
```

---

## 九、测试要求

### test_loop.py

```python
"""
测试 agent_loop 核心逻辑。使用 mock provider 避免真实 API 调用。

必须覆盖的场景：
1. 纯文本响应（无 tool call）→ 循环执行一次就结束
2. 单 tool call → 执行 → 回灌 → LLM 最终文本响应
3. 多 tool call（同一轮）→ 顺序执行 → 回灌所有结果
4. 多轮 tool call → LLM 连续调用 tool 多轮
5. tool 参数验证失败 → 错误信息回灌 → LLM 重试
6. tool 执行异常 → 异常信息回灌 → LLM 处理
7. abort 信号 → 循环中止
8. 未知 tool name → 错误信息回灌
"""
```

### test_tools.py

```python
"""
测试各 tool 的执行逻辑。

bash:
- 正常命令执行（echo hello）
- 非零 exit code
- 超时
- 输出截断（生成 > 2000 行的输出）

read:
- 读取文本文件
- 读取不存在的文件 → 错误
- offset/limit 参数
- 大文件截断

write:
- 写入新文件
- 覆盖已有文件
- 自动创建父目录

edit:
- 正常替换
- old_text 不存在 → 错误
- old_text 出现多次 → 错误
"""
```

### test_session.py

```python
"""
测试 SessionManager。

1. 内存模式：append + build_context 正确重建消息
2. 文件持久化：写入 → 重新打开 → 内容一致
3. branch：分叉后 build_context 只返回新分支的消息
4. tree 结构正确
5. continue_recent 找到最新文件
6. crash recovery：写入一半后重新加载不崩溃
"""
```

---

## 十、pyproject.toml

```toml
[project]
name = "mini-agent"
version = "0.1.0"
description = "Minimal AI coding agent framework inspired by Pi"
requires-python = ">=3.11"
dependencies = [
    "anthropic>=0.40.0",
    "pydantic>=2.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.24",
]

[project.scripts]
mini-agent = "mini_agent.example:main"
```

---

## 十一、实现优先级

按以下顺序实现，每步完成后可独立运行验证：

### Phase 1：能跑通一次完整对话（1-2小时）

1. `types.py` — 所有类型定义
2. `providers/anthropic.py` — 非流式 `complete()` 方法先行
3. `tools/base.py` — Tool 基类 + ToolRegistry
4. `tools/bash.py` — bash 工具
5. `loop.py` — agent_loop()，先用非流式调用
6. `example.py` — 最简入口

验证命令：`python -m mini_agent.example "Run ls -la and tell me what you see"`

### Phase 2：流式 + 完整 tools（2-3小时）

7. `providers/anthropic.py` — 添加 `stream_response()`
8. `loop.py` — 切换到流式调用
9. `tools/read.py`
10. `tools/write.py`
11. `tools/edit.py`

### Phase 3：Agent 封装 + Session（2-3小时）

12. `agent.py` — Agent 类（steer/followUp/subscribe）
13. `session.py` — SessionManager
14. `compact.py` — Compaction

### Phase 4：测试（1-2小时）

15. 全部测试文件

---

## 十二、关键注意事项

### Anthropic API 的特殊要求

1. **messages 必须交替 user/assistant**：不能连续两条同 role 的消息。tool_result 是作为 `role: "user"` 发的。
2. **第一条 message 必须是 user**：不能以 assistant 开头。
3. **tool_result 必须紧跟对应的 tool_use**：如果 assistant 消息包含 tool_use，下一条必须是包含对应 tool_result 的 user 消息。
4. **每个 tool_use 都必须有对应的 tool_result**：不能跳过。

### 输出截断常量

```python
MAX_OUTPUT_LINES = 2000      # 最大输出行数
MAX_OUTPUT_BYTES = 50 * 1024 # 最大输出字节数 (50KB)
DEFAULT_READ_LINES = 2000    # read 工具默认行数
```

### 内部 endpoint 适配

用户可能通过 `base_url` 指向内部 LLM 服务。这些服务通常兼容 Anthropic API 格式，但可能有差异：

1. 可能不支持 streaming → 确保 `complete()` 方法能独立工作
2. 可能需要额外 header（通过 `ProviderConfig.default_headers` 传入）
3. 模型 ID 可能不同 → 不做模型 ID 验证，直接传给 API

### 并发安全

- 一个 Agent 实例同一时间只运行一个 `prompt()`
- steer/followUp 可以从其他线程/协程调用（queue 用 `list` 足够，因为 Python GIL）
- abort_signal 是 `asyncio.Event`，线程安全

---

## 十三、不做什么（明确排除）

以下功能 **不在本项目范围内**，不要实现：

- ❌ 多 provider 支持（只做 Anthropic）
- ❌ MCP 支持
- ❌ TUI / 终端界面（只做 simple print）
- ❌ 模型注册表 / 自动发现
- ❌ OAuth 认证
- ❌ 扩展 / 插件系统
- ❌ Thinking / reasoning trace 支持
- ❌ 图片输入支持（read 工具的图片功能可以暂缓）
- ❌ 跨 provider context 切换
- ❌ Token cost 计算（只做 token count 追踪）
- ❌ 自定义主题
- ❌ 斜杠命令

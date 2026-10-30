"""
Model Context Protocol (MCP) Tool Adapter.

Enables ResearchAgent to seamlessly connect to any external MCP server
(via stdio JSON-RPC or command execution) and expose its tools as native
ResearchAgent BaseTool instances.

Usage::

    # Connect to a local SQLite or filesystem MCP server
    from research_agent.tools.mcp import load_mcp_tools, MCPTool

    tools = await load_mcp_tools(command="npx", args=["-y", "@modelcontextprotocol/server-sqlite"])
    agent = ResearchAgent(tools=tools)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
from typing import Any, Dict, List, Optional

from research_agent.tools.base import BaseTool, ToolResult

logger = logging.getLogger(__name__)


class MCPTool(BaseTool):
    """Wraps an individual tool exported by an external Model Context Protocol server.

    Attributes:
        name: The MCP tool identifier.
        description: Description provided by the MCP server.
        input_schema: JSON Schema defining parameters.
        client: The active :class:`MCPClient` handling JSON-RPC communication.
    """

    def __init__(
        self,
        name: str,
        description: str,
        input_schema: dict[str, Any],
        client: "MCPClient",
    ) -> None:
        self.name = name
        self.description = description
        self._schema_dict = input_schema
        self.client = client

    @property
    def schema(self) -> dict[str, Any]:
        """Convert MCP inputSchema to OpenAI-compatible function definition."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self._schema_dict,
            },
        }

    async def run(self, **kwargs: Any) -> ToolResult:
        """Call the tool on the external MCP server via JSON-RPC."""
        try:
            result = await self.client.call_tool(self.name, kwargs)
            return self._ok(result)
        except Exception as exc:
            logger.error("MCP tool '%s' call failed: %s", self.name, exc)
            return self._fail(f"MCP error: {exc}")


class MCPClient:
    """Asynchronous JSON-RPC client communicating with an MCP server over stdio."""

    def __init__(self, command: str, args: Optional[list[str]] = None, env: Optional[dict[str, str]] = None) -> None:
        self.command = command
        self.args = args or []
        self.env = env or dict(os.environ)
        self.process: Optional[asyncio.subprocess.Process] = None
        self._msg_id = 0
        self._pending: dict[int, asyncio.Future] = {}
        self._reader_task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        """Launch the MCP server process and initialize the connection."""
        if not shutil.which(self.command):
            raise FileNotFoundError(f"MCP server executable '{self.command}' not found on PATH.")

        cmd = [self.command] + self.args
        self.process = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=self.env,
        )

        self._reader_task = asyncio.create_task(self._read_stdout())

        # Send MCP initialize handshake
        init_req = {
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "clientInfo": {"name": "ResearchAgent", "version": "0.1.0"},
            },
        }
        await self._send(init_req)

    def _next_id(self) -> int:
        self._msg_id += 1
        return self._msg_id

    async def _send(self, payload: dict[str, Any]) -> Any:
        """Send a JSON-RPC message and await response."""
        if not self.process or not self.process.stdin:
            raise RuntimeError("MCP process is not running.")

        msg_id = payload.get("id")
        fut = asyncio.get_event_loop().create_future()
        if msg_id is not None:
            self._pending[msg_id] = fut

        line = json.dumps(payload) + "\n"
        self.process.stdin.write(line.encode("utf-8"))
        await self.process.stdin.drain()

        if msg_id is not None:
            return await asyncio.wait_for(fut, timeout=15.0)
        return None

    async def _read_stdout(self) -> None:
        """Background task reading JSON-RPC responses from server stdout."""
        assert self.process and self.process.stdout
        while True:
            line = await self.process.stdout.readline()
            if not line:
                break
            try:
                data = json.loads(line.decode("utf-8").strip())
                msg_id = data.get("id")
                if msg_id in self._pending:
                    fut = self._pending.pop(msg_id)
                    if "error" in data:
                        fut.set_exception(RuntimeError(data["error"].get("message", "Unknown MCP error")))
                    else:
                        fut.set_result(data.get("result"))
            except Exception:
                continue

    async def list_tools(self) -> list[dict[str, Any]]:
        """Query the MCP server for available tools."""
        req = {
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": "tools/list",
            "params": {},
        }
        res = await self._send(req)
        return res.get("tools", []) if isinstance(res, dict) else []

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        """Invoke a tool on the MCP server."""
        req = {
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": "tools/call",
            "params": {"name": name, "arguments": arguments},
        }
        res = await self._send(req)
        if isinstance(res, dict) and "content" in res:
            # Extract text elements
            texts = [c.get("text", "") for c in res["content"] if isinstance(c, dict) and c.get("type") == "text"]
            return "\n".join(texts) if texts else res["content"]
        return res

    async def close(self) -> None:
        """Terminate the MCP server process."""
        if self._reader_task:
            self._reader_task.cancel()
        if self.process:
            try:
                self.process.terminate()
                await self.process.wait()
            except ProcessLookupError:
                pass


async def load_mcp_tools(command: str, args: Optional[list[str]] = None) -> list[BaseTool]:
    """Connect to an external MCP server and return wrapped BaseTool instances.

    Args:
        command: Binary/executable name (e.g. "npx", "uvx", "python").
        args: Command-line arguments to pass to the MCP server.

    Returns:
        List of :class:`MCPTool` instances ready to register with ResearchAgent.
    """
    client = MCPClient(command=command, args=args)
    await client.start()
    tools_data = await client.list_tools()

    mcp_tools = []
    for t in tools_data:
        mcp_tools.append(
            MCPTool(
                name=t.get("name", "unnamed_mcp_tool"),
                description=t.get("description", ""),
                input_schema=t.get("inputSchema", {}),
                client=client,
            )
        )
    return mcp_tools


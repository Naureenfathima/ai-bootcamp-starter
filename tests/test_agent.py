"""
tests/test_agent.py — Unit tests for the agent tools and memory.

STUDENT TODO:
  - Add tests for the full Agent.run() loop (mock the Claude API).
  - Add tests for multi-step tool chains.
  - Add tests for the max_agent_steps safety limit.
"""

import pytest
from app.agent.tools import CalculatorTool, CurrentTimeTool
from app.agent.memory import EpisodicMemory, WorkingMemory


# ── Calculator tool tests ─────────────────────────────────────────────────────
class TestCalculatorTool:
    tool = CalculatorTool()

    def test_addition(self):
        assert self.tool.execute(expression="2 + 3") == "5"

    def test_multiplication(self):
        assert self.tool.execute(expression="6 * 7") == "42"

    def test_sqrt(self):
        result = float(self.tool.execute(expression="sqrt(144)"))
        assert result == pytest.approx(12.0)

    def test_complex_expression(self):
        result = float(self.tool.execute(expression="sqrt(16) * 3 + 2 ** 3"))
        assert result == pytest.approx(20.0)

    def test_division(self):
        result = float(self.tool.execute(expression="10 / 4"))
        assert result == pytest.approx(2.5)

    def test_invalid_expression_returns_error(self):
        result = self.tool.execute(expression="not valid math !!!")
        assert "Error" in result

    def test_cannot_execute_arbitrary_code(self):
        # Should not allow access to builtins
        result = self.tool.execute(expression="__import__('os').system('ls')")
        assert "Error" in result

    def test_tool_schema_has_required_fields(self):
        schema = self.tool.as_claude_tool()
        assert schema["name"] == "calculator"
        assert "description" in schema
        assert "input_schema" in schema


# ── CurrentTimeTool tests ─────────────────────────────────────────────────────
class TestCurrentTimeTool:
    tool = CurrentTimeTool()

    def test_returns_string(self):
        result = self.tool.execute()
        assert isinstance(result, str)

    def test_contains_utc(self):
        result = self.tool.execute()
        assert "UTC" in result

    def test_contains_date_format(self):
        import re
        result = self.tool.execute()
        # Should match YYYY-MM-DD HH:MM:SS UTC
        assert re.match(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} UTC", result)


# ── EpisodicMemory tests ──────────────────────────────────────────────────────
class TestEpisodicMemory:

    def test_add_and_retrieve(self):
        mem = EpisodicMemory()
        mem.add("user", "Hello")
        mem.add("assistant", "Hi there!")
        messages = mem.as_claude_messages()
        assert len(messages) == 2
        assert messages[0] == {"role": "user", "content": "Hello"}
        assert messages[1] == {"role": "assistant", "content": "Hi there!"}

    def test_clear(self):
        mem = EpisodicMemory()
        mem.add("user", "test")
        mem.clear()
        assert len(mem) == 0

    def test_sliding_window(self):
        from app.agent.memory import MAX_HISTORY_MESSAGES
        mem = EpisodicMemory()
        for i in range(MAX_HISTORY_MESSAGES + 5):
            mem.add("user", f"message {i}")
        assert len(mem) == MAX_HISTORY_MESSAGES

    def test_last_n(self):
        mem = EpisodicMemory()
        for i in range(5):
            mem.add("user", f"msg {i}")
        last2 = mem.last_n(2)
        assert len(last2) == 2
        assert last2[-1].content == "msg 4"


# ── WorkingMemory tests ───────────────────────────────────────────────────────
class TestWorkingMemory:

    def test_set_and_get(self):
        wm = WorkingMemory()
        wm.set("key", "value")
        assert wm.get("key") == "value"

    def test_missing_key_returns_default(self):
        wm = WorkingMemory()
        assert wm.get("missing") is None
        assert wm.get("missing", "fallback") == "fallback"

    def test_clear(self):
        wm = WorkingMemory()
        wm.set("x", 1)
        wm.clear()
        assert wm.get("x") is None

    def test_overwrite(self):
        wm = WorkingMemory()
        wm.set("k", "first")
        wm.set("k", "second")
        assert wm.get("k") == "second"

    def test_as_dict(self):
        wm = WorkingMemory()
        wm.set("a", 1)
        wm.set("b", 2)
        d = wm.as_dict()
        assert d == {"a": 1, "b": 2}

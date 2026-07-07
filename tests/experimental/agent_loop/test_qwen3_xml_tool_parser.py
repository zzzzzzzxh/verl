# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import asyncio
import importlib.util
import logging
import sys
import types
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]


def _install_module_stub(name: str, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules[name] = module
    return module


def _identity_trace_op(func):
    return func


_install_module_stub("verl")
_install_module_stub("verl.tools")
_install_module_stub("verl.tools.schemas", OpenAIFunctionToolSchema=object)
_install_module_stub("verl.utils")
_install_module_stub("verl.utils.ray_utils", get_event_loop=asyncio.get_event_loop)
_install_module_stub("verl.utils.rollout_trace", rollout_trace_op=_identity_trace_op)

spec = importlib.util.spec_from_file_location(
    "qwen3_xml_tool_parser_test_module",
    REPO_ROOT / "verl/experimental/agent_loop/tool_parser.py",
)
tool_parser_module = importlib.util.module_from_spec(spec)
assert spec and spec.loader
sys.modules[spec.name] = tool_parser_module
spec.loader.exec_module(tool_parser_module)
Qwen3XMLToolParser = tool_parser_module.Qwen3XMLToolParser


class DummyTokenizer:
    def __init__(self, text: str):
        self.text = text

    def decode(self, response_ids):
        return self.text


def test_qwen3_xml_tool_parser_logs_malformed_model_output(caplog):
    model_output = "thinking...\n<tool_call>\n<function=run_shell>\n<parameter=cmd</tool_call>"
    parser = Qwen3XMLToolParser(DummyTokenizer(model_output))

    with caplog.at_level(logging.ERROR):
        content, tool_calls = asyncio.run(parser.extract_tool_calls([1, 2, 3], tools=[]))

    assert content == model_output
    assert tool_calls == []

    log_text = caplog.text
    assert "Failed to parse XML tool call from model output" in log_text
    assert "Extracted function call candidates" in log_text
    assert "run_shell>" in log_text
    assert "<parameter=cmd" in log_text
    assert "Full model output preview" in log_text
    assert "<parameter=cmd</tool_call>" in log_text

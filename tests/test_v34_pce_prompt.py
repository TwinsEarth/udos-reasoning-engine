"""v3.4.0.dev3 PCE 物理提示词数据包接口测试。"""
import json

import pytest
import torch

from udos import __version__
from udos.pce_format import DemonstrationPrompt, PCEPromptParser
from udos.icm import DemonstrationMemory


def test_prompt_structure():
    p = DemonstrationPrompt("demo-001", kind="collision",
                            adaptation={"source_dof": 4, "target_dof": 6,
                                        "src_freq": 10.0, "dst_freq": 20.0})
    p.add_block(torch.randn(6, 6), torch.randn(6), torch.randn(6))
    p.add_block(torch.randn(6, 6), torch.randn(6), torch.randn(6))
    assert len(p) == 2
    d = p.to_dict()
    assert d["packet"] == "PCE-DemonstrationPrompt/v1"
    assert d["adaptation"]["source_dof"] == 4
    assert len(d["causal_blocks"]) == 2


def test_roundtrip_json():
    p = DemonstrationPrompt("demo-rt", kind="spring",
                            adaptation={"source_dof": 6, "target_dof": 6})
    p.add_block(torch.randn(6, 6), torch.randn(6), torch.randn(6))
    text = p.dumps()
    raw = json.loads(text)
    p2 = PCEPromptParser.from_dict(raw)
    assert p2.prompt_id == "demo-rt"
    assert p2.kind == "spring"
    assert len(p2) == 1
    # 解析为 episode 列表
    eps = PCEPromptParser.load_episodes(p2)
    assert len(eps) == 1
    mem = DemonstrationMemory()
    mem.register(eps[0])
    assert mem.size == 1


def test_unknown_packet_version_rejected():
    with pytest.raises(ValueError):
        PCEPromptParser.from_dict({"packet": "wrong/v9", "prompt_id": "x"})


def test_non_object_rejected():
    with pytest.raises(ValueError):
        PCEPromptParser.from_dict([1, 2, 3])


def test_empty_prompt_loads_zero_episodes():
    p = DemonstrationPrompt("empty")
    assert len(p) == 0
    assert PCEPromptParser.load_episodes(p) == []


def test_add_block_bad_shape():
    p = DemonstrationPrompt("bad")
    with pytest.raises(ValueError):
        p.add_block(torch.randn(6, 5), torch.randn(6), torch.randn(6))


def test_empty_prompt_id_rejected():
    with pytest.raises(ValueError):
        DemonstrationPrompt("")


def test_version():
    assert __version__ == "5.5.5"

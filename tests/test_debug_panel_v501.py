"""v5.0.1 P0 补丁: DebugPanel 覆盖率单测 (去口令后逻辑)。"""
import os

import torch


def _panel(level=1, enabled=None):
    from udos.debug import DebugPanel
    return DebugPanel(level=level, enabled=enabled)


def test_default_off_without_env(monkeypatch):
    monkeypatch.delenv("UDOS_DEBUG", raising=False)
    panel = _panel(level=2)
    assert panel.enabled is False
    x = torch.zeros(4)
    panel.shape("x", x)          # 静默不抛
    assert panel.dump() == []


def test_env_on(monkeypatch):
    monkeypatch.setenv("UDOS_DEBUG", "1")
    panel = _panel(level=2)
    assert panel.enabled is True
    panel.shape("x", torch.zeros(4))
    assert panel.dump() != []


def test_explicit_enabled_override(monkeypatch):
    monkeypatch.setenv("UDOS_DEBUG", "0")
    assert _panel(level=1, enabled=True).enabled is True
    monkeypatch.setenv("UDOS_DEBUG", "1")
    assert _panel(level=1, enabled=False).enabled is False


def test_level_clamp_and_section(monkeypatch):
    monkeypatch.setenv("UDOS_DEBUG", "0")
    panel = _panel(level=5, enabled=True)
    assert panel.level == 3
    with panel.section("op"):
        panel.shape("x", torch.ones(2, 2))
    assert any("op" in r for r in panel.dump())


def test_trace_kv_stats_dump(monkeypatch):
    monkeypatch.setenv("UDOS_DEBUG", "0")
    panel = _panel(level=3, enabled=True)
    panel.trace("cert", [0.1, 0.2, 0.3])
    panel.trace("cert2", torch.tensor([1.0, 2.0]))
    panel.kv("sync", 0.9)
    st = panel.stats("w", torch.randn(4, 4))
    assert "mean" in st
    panel.trace("e", [])          # 空 trace 不抛
    assert isinstance(panel.dump(), list)


def test_global_panel(monkeypatch):
    monkeypatch.setenv("UDOS_DEBUG", "0")
    p = _panel(level=1, enabled=False)
    from udos.debug import DebugPanel
    assert DebugPanel.global_panel() is p


def test_no_password_param():
    import inspect
    from udos.debug import DebugPanel
    sig = inspect.signature(DebugPanel.__init__)
    assert "password" not in sig.parameters

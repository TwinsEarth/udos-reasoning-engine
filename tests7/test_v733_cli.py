"""v7.3.3 跨平台命令行工具 udos7.cli 的契约测试。"""
import json
import os
import socket
import sys
from pathlib import Path

import pytest

from udos7 import __version__
from udos7 import cli

ROOT = Path(__file__).resolve().parents[1]


def _free_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def test_version(capsys):
    assert cli.main(["version"]) == 0
    assert capsys.readouterr().out.strip() == __version__


def test_dash_dash_version(capsys):
    assert cli.main(["--version"]) == 0
    assert __version__ in capsys.readouterr().out


def test_no_args_prints_help(capsys):
    assert cli.main([]) == 0
    out = capsys.readouterr().out
    for name in ("serve", "test", "demo", "health", "predict"):
        assert name in out


def test_info_is_valid_json(capsys):
    assert cli.main(["info"]) == 0
    d = json.loads(capsys.readouterr().out)
    assert d["version"] == __version__
    assert d["platform"] == sys.platform
    assert "repo_root" in d


def test_health_unreachable_returns_1(capsys):
    port = _free_port()
    rc = cli.main(["health", "--port", str(port), "--timeout", "0.5"])
    assert rc == 1
    assert "无法连接" in capsys.readouterr().err


def test_serve_dispatches(monkeypatch):
    seen = {}

    def fake_serve(host, port, checkpoint):
        seen.update(host=host, port=port, checkpoint=checkpoint)

    monkeypatch.setattr("udos7.server.serve", fake_serve)
    assert cli.main(["serve", "--port", "8799", "--checkpoint", "x.pt"]) == 0
    assert seen == {"host": "127.0.0.1", "port": 8799, "checkpoint": "x.pt"}


def test_serve_auto_port_when_occupied(monkeypatch):
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.bind(("127.0.0.1", 0))
    srv.listen()
    busy = srv.getsockname()[1]
    seen = {}
    monkeypatch.setattr("udos7.server.serve",
                        lambda host, port, checkpoint: seen.update(port=port))
    rc = cli.main(["serve", "--port", str(busy), "--auto-port"])
    srv.close()
    assert rc == 0
    assert seen["port"] != busy  # 自动换了空闲端口


def test_test_dispatch(monkeypatch):
    seen = {}
    monkeypatch.setattr(cli, "_run_pytest",
                        lambda suite, extra: seen.update(suite=suite, extra=extra) or 0)
    assert cli.main(["test", "--suite", "all"]) == 0
    assert seen["suite"] == "all"


def test_demo_list(capsys):
    assert cli.main(["demo", "--list"]) == 0
    out = capsys.readouterr().out
    assert "obs-pro" in out and "legion" in out


def test_demo_unknown_returns_2(capsys):
    assert cli.main(["demo", "nope"]) == 2


def test_demo_obs_pro_end_to_end():
    # 零第三方可观测演示，真实跑通并产报告
    assert cli.main(["demo", "obs-pro"]) == 0
    assert (ROOT / "reports7" / "observability_pro_demo.json").exists()


def test_predict_bad_json_returns_2(tmp_path, capsys):
    f = tmp_path / "w.json"
    f.write_text("{ not json", encoding="utf-8")
    assert cli.main(["predict", str(f)]) == 2
    assert "解析失败" in capsys.readouterr().err


def test_native_wrappers_present():
    posix = ROOT / "bin" / "udos"
    win = ROOT / "bin" / "udos.bat"
    assert posix.exists() and os.access(posix, os.X_OK)
    p = posix.read_text(encoding="utf-8")
    assert "udos7.cli" in p and "udos-env.py" in p  # 走共享环境引导
    w = win.read_text(encoding="utf-8")
    assert "udos7.cli" in w and "udos-env.py" in w
    assert (ROOT / "bin" / "udos-env.py").exists()


def test_console_script_registered():
    s = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'udos = "udos7.cli:main"' in s

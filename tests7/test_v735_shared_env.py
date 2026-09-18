"""v7.3.5 共享虚拟环境引导 bin/udos-env.py 的契约测试（不联网、不真装 torch）。"""
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("udos_env", ROOT / "bin" / "udos-env.py")
env = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(env)


def test_shared_dir_override(tmp_path):
    custom = tmp_path / "myvenv"
    d = env.shared_venv_dir(env={"UDOS_VENV": str(custom)}, home=tmp_path)
    assert d == custom


def test_shared_dir_by_platform(tmp_path):
    win = env.shared_venv_dir(env={}, home=tmp_path, platform_sys="win32")
    assert win == tmp_path / ".udos" / "venv"
    mac = env.shared_venv_dir(env={}, home=tmp_path, platform_sys="darwin")
    assert mac == tmp_path / ".udos" / "venv"


def test_venv_python_layout(tmp_path):
    assert env.venv_python(tmp_path / "v", "win32").name == "python.exe"
    assert env.venv_python(tmp_path / "v", "darwin") == tmp_path / "v" / "bin" / "python"


def test_hash_deterministic():
    h1, h2 = env.requirements_hash(), env.requirements_hash()
    assert h1 == h2 and len(h1) == 16


def test_legacy_local_venv_detected_windows(tmp_path):
    # 旧版在仓库内建过 .venv-win：应被沿用，避免再下一遍 torch
    py = tmp_path / ".venv-win" / "Scripts" / "python.exe"
    py.parent.mkdir(parents=True)
    py.write_text("", encoding="utf-8")
    got = env.legacy_local_venv(tmp_path, "win32")
    assert got == tmp_path / ".venv-win"


def test_legacy_local_venv_detected_posix(tmp_path):
    py = tmp_path / ".venv-udos" / "bin" / "python"
    py.parent.mkdir(parents=True)
    py.write_text("", encoding="utf-8")
    assert env.legacy_local_venv(tmp_path, "darwin") == tmp_path / ".venv-udos"
    assert env.legacy_local_venv(tmp_path, "win32") is None


def test_already_provisioned_requires_marker_and_imports(tmp_path, monkeypatch):
    h = env.requirements_hash()
    # 无 python / 无 marker → False
    assert env.already_provisioned(tmp_path / "nope", h, "linux") is False
    vdir = tmp_path / "v"
    (vdir / "bin").mkdir(parents=True)
    py = vdir / "bin" / "python"
    py.write_text("", encoding="utf-8")
    monkeypatch.setattr(env, "imports_ok", lambda p: True)
    # 有 python 无 marker → False
    assert env.already_provisioned(vdir, h, "linux") is False
    env.marker_path(vdir).write_text(json.dumps({"requirements_hash": h}), encoding="utf-8")
    assert env.already_provisioned(vdir, h, "linux") is True
    # 哈希不匹配（依赖升级）→ False，触发增量安装
    assert env.already_provisioned(vdir, "deadbeefdeadbeef", "linux") is False


def test_already_provisioned_false_when_imports_broken(tmp_path, monkeypatch):
    vdir = tmp_path / "v"
    (vdir / "bin").mkdir(parents=True)
    (vdir / "bin" / "python").write_text("", encoding="utf-8")
    env.marker_path(vdir).write_text(
        json.dumps({"requirements_hash": env.requirements_hash()}), encoding="utf-8")
    monkeypatch.setattr(env, "imports_ok", lambda p: False)
    assert env.already_provisioned(vdir, env.requirements_hash(), "linux") is False


def test_info_json_with_override(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("UDOS_VENV", str(tmp_path / "custom"))
    rc = env.main(["info"])
    assert rc == 0
    d = json.loads(capsys.readouterr().out)
    assert d["venv"].endswith("custom") and d["provisioned"] is False
    assert d["requirements_hash"] == env.requirements_hash()

"""v7.3.6 双击菜单与根目录便捷入口的契约测试。"""
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_windows_menu_present_and_wired():
    f = ROOT / "UDOS-命令行菜单.bat"
    assert f.exists()
    s = f.read_text(encoding="utf-8")
    assert r"bin\udos.bat" in s
    for token in ("info", "test", "serve", "health", "demo obs-pro", "demo legion"):
        assert token in s


def test_mac_menu_present_executable_and_valid_bash():
    f = ROOT / "UDOS-命令行菜单.command"
    assert f.exists() and os.access(f, os.X_OK)
    s = f.read_text(encoding="utf-8")
    assert "./bin/udos" in s
    for token in ("info", "test", "serve", "health", "obs-pro", "legion"):
        assert token in s
    r = subprocess.run(["bash", "-n", str(f)], capture_output=True)
    assert r.returncode == 0, r.stderr.decode()


def test_root_convenience_wrappers():
    win = ROOT / "udos.bat"
    posix = ROOT / "udos.sh"
    assert win.exists() and r"bin\udos.bat" in win.read_text(encoding="utf-8")
    assert posix.exists() and os.access(posix, os.X_OK)
    assert "/bin/udos" in posix.read_text(encoding="utf-8")
    assert subprocess.run(["bash", "-n", str(posix)], capture_output=True).returncode == 0


def test_root_posix_wrapper_runs_version(tmp_path):
    import importlib.util
    import json
    import venv
    # 建一个带系统站点的共享环境并直接写“已就绪”标记，避免测试触发联网安装
    v = tmp_path / "sh"
    venv.EnvBuilder(system_site_packages=True).create(str(v))
    spec = importlib.util.spec_from_file_location("udos_env", ROOT / "bin" / "udos-env.py")
    ue = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ue)
    ue.marker_path(v).write_text(
        json.dumps({"requirements_hash": ue.requirements_hash()}), encoding="utf-8")
    r = subprocess.run(
        ["bash", str(ROOT / "udos.sh"), "version"],
        capture_output=True, text=True, timeout=60,
        env={**os.environ, "PYTHONPATH": str(ROOT), "UDOS_VENV": str(v)})
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip().splitlines()[-1].count(".") == 2  # x.y.z

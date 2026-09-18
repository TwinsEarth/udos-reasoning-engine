#!/usr/bin/env python3
"""UDOS 跨平台共享环境引导（macOS / Linux / Windows）。

目标：torch 等重依赖**只安装一次**，多个 UDOS 版本/解压目录共享同一个虚拟环境，
升级版本默认不再重复下载安装；仅当依赖清单哈希变化时才增量安装。

环境位置解析顺序：
  1. 环境变量 UDOS_VENV（显式指定，优先级最高）
  2. 用户主目录共享环境：
       Windows : %USERPROFILE%\\.udos\\venv
       其它    : $HOME/.udos/venv
  3. 兼容旧版：仓库内 .venv-udos / .venv-win（若已存在则沿用，不强迫迁移）

用法（由 bin/udos、bin/udos.bat、一键脚本内部调用）：
  python udos-env.py ensure     # 确保环境就绪，stdout 最后一行打印该环境的 python 路径
  python udos-env.py path       # 仅打印将使用的环境目录
  python udos-env.py info       # 打印环境/哈希/是否已就绪（JSON）
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # 仓库根
# 变更引导逻辑或依赖清单时提升此版本，会触发一次性增量安装
BOOTSTRAP_VERSION = "1"
TORCH_SPEC = "torch==2.14.0"
# torch 之外的轻依赖
REQUIREMENTS = ["numpy>=1.26", "huggingface_hub>=0.20", "pytest>=7"]
IMPORT_CHECK = ["torch", "numpy", "huggingface_hub", "pytest"]
CPU_TORCH_INDEX = "https://download.pytorch.org/whl/cpu"
TUNA_INDEX = "https://pypi.tuna.tsinghua.edu.cn/simple"
MARKER_NAME = "udos_provision.json"


def shared_venv_dir(env: dict | None = None, home: Path | None = None,
                    platform_sys: str | None = None) -> Path:
    env = env if env is not None else os.environ
    home = home if home is not None else Path.home()
    platform_sys = platform_sys or sys.platform
    if env.get("UDOS_VENV"):
        return Path(env["UDOS_VENV"]).expanduser()
    return home / ".udos" / "venv"


def legacy_local_venv(root: Path, platform_sys: str) -> Path | None:
    """旧版本在仓库内建过环境则继续沿用，避免再下一遍 torch。"""
    candidates = ([root / ".venv-win", root / ".venv-udos", root / ".venv"]
                  if platform_sys.startswith("win")
                  else [root / ".venv-udos", root / ".venv", root / ".venv-win"])
    for c in candidates:
        if venv_python(c, platform_sys).exists():
            return c
    return None


def venv_python(venv_dir: Path, platform_sys: str) -> Path:
    if platform_sys.startswith("win"):
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def requirements_hash() -> str:
    blob = "\n".join([BOOTSTRAP_VERSION, TORCH_SPEC] + REQUIREMENTS)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def marker_path(venv_dir: Path) -> Path:
    return venv_dir / MARKER_NAME


def imports_ok(py: Path) -> bool:
    code = "import " + ",".join(IMPORT_CHECK)
    try:
        r = subprocess.run([str(py), "-c", code], capture_output=True, timeout=120)
        return r.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def already_provisioned(venv_dir: Path, h: str, platform_sys: str) -> bool:
    mp = marker_path(venv_dir)
    py = venv_python(venv_dir, platform_sys)
    if not py.exists() or not mp.exists():
        return False
    try:
        data = json.loads(mp.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return data.get("requirements_hash") == h and imports_ok(py)


def _run(cmd: list[str]) -> bool:
    try:
        return subprocess.run(cmd, check=False).returncode == 0
    except OSError:
        return False


def provision(venv_dir: Path, platform_sys: str) -> None:
    py = venv_python(venv_dir, platform_sys)
    if not py.exists():
        print(f"[UDOS] 首次运行，创建共享虚拟环境：{venv_dir}", flush=True)
        venv_dir.parent.mkdir(parents=True, exist_ok=True)
        venv.EnvBuilder(with_pip=True, clear=False).create(str(venv_dir))
    print("[UDOS] 升级 pip …", flush=True)
    _run([str(py), "-m", "pip", "install", "--upgrade", "pip"])

    is_win = platform_sys.startswith("win")
    is_linux = platform_sys.startswith("linux")
    # torch：Linux/Windows 走 CPU 专用索引（体积小、无 CUDA）；macOS 走 PyPI 官方 wheel（含 MPS）
    if not _import_torch_ok(py):
        if is_win or is_linux:
            ok = _run([str(py), "-m", "pip", "install", TORCH_SPEC,
                       "--index-url", CPU_TORCH_INDEX,
                       "--extra-index-url", "https://pypi.org/simple"])
            if not ok:  # 回退官方源 / 清华源
                _run([str(py), "-m", "pip", "install", TORCH_SPEC]) or \
                    _run([str(py), "-m", "pip", "install", TORCH_SPEC, "-i", TUNA_INDEX])
        else:
            _run([str(py), "-m", "pip", "install", TORCH_SPEC]) or \
                _run([str(py), "-m", "pip", "install", TORCH_SPEC, "-i", TUNA_INDEX])

    # 轻依赖：官方源失败回退清华
    if not _run([str(py), "-m", "pip", "install", *REQUIREMENTS]):
        _run([str(py), "-m", "pip", "install", *REQUIREMENTS, "-i", TUNA_INDEX])

    if not imports_ok(py):
        print("[UDOS][警告] 依赖自检未完全通过，可手动在该环境内 pip install。", flush=True)
        return
    marker_path(venv_dir).write_text(json.dumps({
        "requirements_hash": requirements_hash(),
        "bootstrap_version": BOOTSTRAP_VERSION,
        "torch_spec": TORCH_SPEC,
        "python": sys.version.split()[0],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[UDOS] 共享环境就绪：{venv_dir}（后续版本直接复用，无需重复安装）", flush=True)


def _import_torch_ok(py: Path) -> bool:
    try:
        return subprocess.run([str(py), "-c", "import torch"],
                              capture_output=True, timeout=60).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def resolve_venv() -> Path:
    shared = shared_venv_dir()
    legacy = legacy_local_venv(ROOT, sys.platform)
    # 显式指定 > 已就绪共享环境 > 旧的仓库内环境（沿用以免重复下载）> 共享环境（新建）
    if os.environ.get("UDOS_VENV"):
        return shared
    mp = marker_path(shared)
    if venv_python(shared, sys.platform).exists() and mp.exists():
        return shared
    if legacy is not None:
        return legacy
    return shared


def ensure() -> Path:
    vdir = resolve_venv()
    h = requirements_hash()
    if not already_provisioned(vdir, h, sys.platform):
        provision(vdir, sys.platform)
    return venv_python(vdir, sys.platform)


def main(argv: list[str]) -> int:
    cmd = argv[0] if argv else "ensure"
    if cmd == "path":
        print(resolve_venv())
        return 0
    if cmd == "info":
        vdir = resolve_venv()
        h = requirements_hash()
        print(json.dumps({
            "venv": str(vdir),
            "python": str(venv_python(vdir, sys.platform)),
            "requirements_hash": h,
            "provisioned": already_provisioned(vdir, h, sys.platform),
            "override": os.environ.get("UDOS_VENV"),
        }, ensure_ascii=False, indent=2))
        return 0
    if cmd == "ensure":
        py = ensure()
        # 最后一行输出 python 路径，供包装脚本捕获；前面的进度信息在 stderr 更稳妥，
        # 但 Windows for /f 只截 stdout，故进度也走 stdout，包装器取最后一行。
        print(str(py))
        return 0
    print(f"未知子命令：{cmd}（可用 ensure/path/info）", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

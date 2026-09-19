"""UDOS Engine 跨平台命令行工具（macOS / Linux / Windows 通用）。

用法：
    python -m udos7.cli <command> [options]
    udos  <command> ...        # pip install -e . 后；或用仓库根的 udos / udos.bat 包装

子命令：
    version     打印版本
    info        打印运行环境/资源/检查点信息（JSON）
    serve       启动 v7 HTTP 引擎服务
    test        运行回归测试（tests7 / tests / all）
    train       训练 v7 预测器（透传给 scripts7/train_v7.py）
    verify      运行 v7 验证与证据报告（scripts7/verify_v7.py）
    demo        运行内置演示：obs / obs-pro / legion / scaling / embodied / spatial / splat / matrix
    health      探测运行中引擎的健康检查
    predict     向运行中的引擎提交一段窗口 JSON 做预测
    metrics     拉取运行中引擎的指标 JSON

零第三方强依赖：health/metrics/predict 仅用标准库 urllib；torch 缺失时 info 如实标注。
"""
from __future__ import annotations

import argparse
import json
import os
import runpy
import socket
import sys
import urllib.error
import urllib.request
from pathlib import Path

from . import __version__

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts7"
DEFAULT_PORT = 8777


# --------------------------------------------------------------------------
# 子命令实现
# --------------------------------------------------------------------------
def _cmd_version(_args) -> int:
    print(__version__)
    return 0


def _cmd_info(_args) -> int:
    info = {
        "name": "udos-reasoning-engine",
        "version": __version__,
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "platform": sys.platform,
        "cwd": os.getcwd(),
        "repo_root": str(ROOT),
    }
    try:
        import torch  # noqa
        info["torch"] = torch.__version__
        info["threads"] = torch.get_num_threads()
        cuda = bool(getattr(torch, "cuda", None) and torch.cuda.is_available())
        info["cuda_available"] = cuda
        if cuda:
            info["cuda_device"] = torch.cuda.get_device_name(0)
        mps = bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
        info["mps_available"] = mps
    except Exception as e:  # 没装 torch 也要能给信息
        info["torch"] = None
        info["torch_error"] = f"{type(e).__name__}: {e}"
    try:  # 可观测资源采集本身已跨平台
        from .observability import resource_usage
        info["resource"] = resource_usage()
    except Exception as e:
        info["resource"] = None
        info["resource_error"] = f"{type(e).__name__}: {e}"
    ckpt = ROOT / "checkpoints7" / "worldmodel_v7.0.3.pt"
    info["default_checkpoint"] = str(ckpt)
    info["checkpoint_exists"] = ckpt.exists()
    print(json.dumps(info, ensure_ascii=False, indent=2))
    return 0


def _cmd_serve(args) -> int:
    from .server import serve
    host = args.host
    port = _free_port_or_given(host, args.port) if args.auto_port else args.port
    print(f"启动 UDOS v{__version__} 服务：http://{host}:{port}/api/v7/health （Ctrl+C 停止）")
    serve(host=host, port=port, checkpoint=args.checkpoint)
    return 0


def _free_port_or_given(host: str, preferred: int) -> int:
    """首选端口被占用时自动找一个空闲端口，避免直接崩溃。"""
    if not _port_in_use(host, preferred):
        return preferred
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return s.getsockname()[1]


def _port_in_use(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.3)
        return s.connect_ex((host, port)) == 0


def _run_pytest(suite: str, extra: list[str]) -> int:
    try:
        import pytest
    except ImportError:
        print("未安装 pytest，请先：pip install pytest", file=sys.stderr)
        return 2
    targets = {
        "tests7": [str(ROOT / "tests7")],
        "tests": [str(ROOT / "tests")],
        "all": [str(ROOT / "tests7"), str(ROOT / "tests")],
    }
    args = targets.get(suite)
    if args is None:
        print(f"未知测试套件：{suite}（可选 tests7/tests/all）", file=sys.stderr)
        return 2
    return int(pytest.main(args + ["-q"] + extra))


def _cmd_test(args) -> int:
    return _run_pytest(args.suite, args.pytest_args)


def _run_script(filename: str, argv: list[str]) -> int:
    path = SCRIPTS / filename
    if not path.exists():
        print(f"脚本不存在：{path}", file=sys.stderr)
        return 2
    old_argv = sys.argv
    sys.argv = [str(path)] + argv
    cwd = os.getcwd()
    os.chdir(ROOT)  # 脚本内相对路径（reports7/checkpoints7）以仓库根为基准
    try:
        runpy.run_path(str(path), run_name="__main__")
        return 0
    except SystemExit as e:  # 脚本可能 sys.exit
        code = e.code
        return int(code) if isinstance(code, int) else (0 if code is None else 1)
    finally:
        sys.argv = old_argv
        os.chdir(cwd)


def _cmd_train(args) -> int:
    return _run_script("train_v7.py", args.script_args)


def _cmd_verify(args) -> int:
    return _run_script("verify_v7.py", args.script_args)


_DEMOS = {
    "obs": "observability_demo.py",
    "obs-pro": "observability_pro_demo.py",
    "legion": "legion_demo.py",
    "embodied": "embodied_hybrid_demo.py",
    "spatial": "spatial_novelview_demo.py",
    "splat": "splat_transfer_demo.py",
    "flywheel": "egodata_flywheel_demo.py",
    "matrix": "matrix_demo.py",
    "scaling": "agent_scaling_bench.py",
}


def _cmd_demo(args) -> int:
    if args.list:
        for k, v in _DEMOS.items():
            print(f"{k:8s} {v}")
        return 0
    if args.name not in _DEMOS:
        print(f"未知演示：{args.name}；可用：{', '.join(_DEMOS)} 或用 --list", file=sys.stderr)
        return 2
    return _run_script(_DEMOS[args.name], args.script_args)


def _base_url(args) -> str:
    return args.url.rstrip("/") if args.url else f"http://{args.host}:{args.port}"


def _http_get(url: str, timeout: float):
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.status, resp.read().decode("utf-8")


def _http_post(url: str, payload: dict, timeout: float):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST",
                                 headers={"Content-Type": "application/json",
                                          "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.status, resp.read().decode("utf-8")


def _cmd_health(args) -> int:
    url = _base_url(args) + "/api/v7/health"
    try:
        status, body = _http_get(url, args.timeout)
    except (urllib.error.URLError, OSError) as e:
        print(f"无法连接引擎（{url}）：{e}", file=sys.stderr)
        return 1
    print(body)
    return 0 if status == 200 else 1


def _cmd_metrics(args) -> int:
    url = _base_url(args) + "/api/v7/metrics"
    try:
        status, body = _http_get(url, args.timeout)
    except (urllib.error.URLError, OSError) as e:
        print(f"无法连接引擎（{url}）：{e}", file=sys.stderr)
        return 1
    if args.pretty:
        try:
            print(json.dumps(json.loads(body), ensure_ascii=False, indent=2))
        except json.JSONDecodeError:
            print(body)
    else:
        print(body)
    return 0 if status == 200 else 1


def _cmd_predict(args) -> int:
    raw = Path(args.file).read_text(encoding="utf-8") if args.file != "-" else sys.stdin.read()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"窗口 JSON 解析失败：{e}", file=sys.stderr)
        return 2
    url = _base_url(args) + "/api/v7/predict"
    try:
        status, body = _http_post(url, payload, args.timeout)
    except (urllib.error.URLError, OSError) as e:
        print(f"无法连接引擎（{url}）：{e}\n请先运行：udos serve", file=sys.stderr)
        return 1
    print(json.dumps(json.loads(body), ensure_ascii=False, indent=2) if args.pretty else body)
    return 0 if status == 200 else 1


# --------------------------------------------------------------------------
# 参数解析
# --------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="udos",
        description="UDOS 推演引擎跨平台命令行工具（macOS/Linux/Windows）")
    p.add_argument("--version", action="version", version=f"udos {__version__}")
    sub = p.add_subparsers(dest="command", metavar="<command>")

    sub.add_parser("version", help="打印版本号").set_defaults(func=_cmd_version)

    sub.add_parser("info", help="打印环境/资源/检查点信息（JSON）").set_defaults(func=_cmd_info)

    sp = sub.add_parser("serve", help="启动 v7 HTTP 引擎服务")
    sp.add_argument("--host", default="127.0.0.1")
    sp.add_argument("--port", type=int, default=DEFAULT_PORT)
    sp.add_argument("--checkpoint", default=None, help="模型检查点路径")
    sp.add_argument("--auto-port", action="store_true", help="端口被占用时自动换空闲端口")
    sp.set_defaults(func=_cmd_serve)

    st = sub.add_parser("test", help="运行回归测试")
    st.add_argument("--suite", choices=["tests7", "tests", "all"], default="tests7")
    st.add_argument("pytest_args", nargs=argparse.REMAINDER,
                    help="透传给 pytest 的参数（需以 -- 引导）")
    st.set_defaults(func=_cmd_test)

    strn = sub.add_parser("train", help="训练 v7 预测器（scripts7/train_v7.py）")
    strn.add_argument("script_args", nargs=argparse.REMAINDER)
    strn.set_defaults(func=_cmd_train)

    sv = sub.add_parser("verify", help="运行 v7 验证与证据报告")
    sv.add_argument("script_args", nargs=argparse.REMAINDER)
    sv.set_defaults(func=_cmd_verify)

    sd = sub.add_parser("demo", help="运行内置演示")
    sd.add_argument("name", nargs="?", choices=list(_DEMOS), help="演示名")
    sd.add_argument("--list", action="store_true", help="列出全部演示")
    sd.add_argument("script_args", nargs=argparse.REMAINDER)
    sd.set_defaults(func=_cmd_demo)

    def _net(parser_name, help_text):
        x = sub.add_parser(parser_name, help=help_text)
        x.add_argument("--host", default="127.0.0.1")
        x.add_argument("--port", type=int, default=DEFAULT_PORT)
        x.add_argument("--url", default=None, help="完整基址，给出后覆盖 host/port")
        x.add_argument("--timeout", type=float, default=5.0)
        return x

    sh = _net("health", "探测引擎健康检查")
    sh.set_defaults(func=_cmd_health)

    sm = _net("metrics", "拉取引擎指标 JSON")
    sm.add_argument("--pretty", action="store_true")
    sm.set_defaults(func=_cmd_metrics)

    spd = _net("predict", "读取窗口 JSON 并向运行中引擎请求预测")
    spd.add_argument("file", help="请求体 JSON 文件路径，- 表示标准输入")
    spd.add_argument("--pretty", action="store_true")
    spd.set_defaults(func=_cmd_predict)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as e:
        # --help/--version 为 0；参数或选择非法为 2。作为库调用时返回退出码而非抛出
        code = e.code
        return int(code) if isinstance(code, int) else (0 if code is None else 2)
    if not getattr(args, "func", None):
        parser.print_help()
        return 0
    # REMAINDER 常把分隔符 -- 带进来，统一剔除首项
    for key in ("pytest_args", "script_args"):
        if hasattr(args, key):
            vals = getattr(args, key)
            if vals and vals[0] == "--":
                setattr(args, key, vals[1:])
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())

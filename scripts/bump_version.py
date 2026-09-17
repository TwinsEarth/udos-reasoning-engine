#!/usr/bin/env python3
"""
UDOS 版本号批量升级工具
======================
用法:
    python3 scripts/bump_version.py <new>                 # 自动读取当前版本
    python3 scripts/bump_version.py <old> <new>           # 兼容旧用法

同步更新 (运行时/打包/容器/看板/契约测试, 全部必须一致):
    udos/__init__.py        标题 docstring(vX.Y.Z) 与 __version__ = "X.Y.Z"
    pyproject.toml          version = "X.Y.Z"
    Dockerfile              org.opencontainers.image.version="X.Y.Z"
    docker-compose.yml      image udos-reasoning-engine:X.Y.Z
    Makefile(若存在)        image tag
    web/dashboard_data.json "version": "X.Y.Z" (只改版本, 不动 tests 等历史快照计数)
    web/udos_console.html   <title>/角标 vX.Y.Z 与内嵌 "version"
    tests/test_*.py         assert __version__ == "X.Y.Z"

刻意不碰 (历史快照/变更日志, 不属于"当前版本"):
    udos/finesim 与 tests/test_finesim_v543.py 的 v5.4.3 历史标注、
    CHANGELOG/RELEASE_NOTES/VERIFICATION、benchmarks/、checkpoints/、docs/。

收尾做两处校验: __init__ 版本确为 new; tests 中不得残留旧断言。
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INIT = ROOT / "udos" / "__init__.py"
VERSION_RE = re.compile(r'__version__\s*=\s*"(\d+\.\d+\.\d+)"')
SEMVER = re.compile(r"^\d+\.\d+\.\d+$")


def current_version() -> str:
    m = VERSION_RE.search(INIT.read_text(encoding="utf-8"))
    if not m:
        sys.exit("无法从 udos/__init__.py 解析 __version__")
    return m.group(1)


def replace_in_file(path: Path, old: str, new: str, counts, required=False):
    if not path.exists():
        if required:
            sys.exit(f"[终止] 必需落点缺失: {path.relative_to(ROOT)}")
        return False
    text = path.read_text(encoding="utf-8")
    n = text.count(old)
    if n:
        path.write_text(text.replace(old, new), encoding="utf-8")
        counts[path.relative_to(ROOT).as_posix()] = counts.get(
            path.relative_to(ROOT).as_posix(), 0) + n
        return True
    if required:
        sys.exit(f"[终止] {path.relative_to(ROOT)} 未找到 {old!r}")
    return False


def main():
    args = sys.argv[1:]
    if len(args) == 1:
        new = args[0]
        old = current_version()
    elif len(args) == 2:
        old, new = args
        if current_version() != old:
            sys.exit(f"参数旧版本 {old} 与当前版本 {current_version()} 不一致")
    else:
        sys.exit("用法: bump_version.py <new> 或 bump_version.py <old> <new>")
    if not SEMVER.match(new):
        sys.exit(f"新版本 {new} 不符合 X.Y.Z")
    if old == new:
        sys.exit(f"当前版本已是 {new}, 无需 bump")

    print(f"bump {old} -> {new}")
    counts: dict = {}

    # 1) 运行时版本源
    replace_in_file(INIT, f"v{old}", f"v{new}", counts, required=True)
    replace_in_file(INIT, f'__version__ = "{old}"', f'__version__ = "{new}"',
                    counts, required=True)
    # 2) 打包 / 容器
    replace_in_file(ROOT / "pyproject.toml", f'version = "{old}"',
                    f'version = "{new}"', counts, required=True)
    replace_in_file(ROOT / "Dockerfile",
                    f'org.opencontainers.image.version="{old}"',
                    f'org.opencontainers.image.version="{new}"', counts,
                    required=True)
    replace_in_file(ROOT / "docker-compose.yml",
                    f"udos-reasoning-engine:{old}",
                    f"udos-reasoning-engine:{new}", counts, required=True)
    replace_in_file(ROOT / "Makefile", f"udos-reasoning-engine:{old}",
                    f"udos-reasoning-engine:{new}", counts)
    # 3) 看板 (只改版本字段; 计数快照留给收口时 scripts/build_console_data.py)
    replace_in_file(ROOT / "web" / "dashboard_data.json",
                    f'"version": "{old}"', f'"version": "{new}"', counts)
    replace_in_file(ROOT / "web" / "udos_console.html", f"v{old}", f"v{new}",
                    counts)
    replace_in_file(ROOT / "web" / "udos_console.html",
                    f'"version": "{old}"', f'"version": "{new}"', counts)
    # 4) tests 版本契约断言 (只匹配 __version__ == "X.Y.Z")
    n_files = 0
    for tp in sorted((ROOT / "tests").glob("test_*.py")):
        text = tp.read_text(encoding="utf-8")
        old_assert = f'__version__ == "{old}"'
        if old_assert in text:
            tp.write_text(text.replace(old_assert, f'__version__ == "{new}"'),
                          encoding="utf-8")
            n_files += 1
    print(f"  tests: {n_files} 个文件的版本断言")
    for k, v in sorted(counts.items()):
        print(f"  {k}: {v} 处")

    # 5) 收尾校验
    if current_version() != new:
        sys.exit(f"[终止] bump 后 __version__={current_version()} != {new}")
    leftover = [tp.name for tp in (ROOT / "tests").glob("test_*.py")
                if f'__version__ == "{old}"' in tp.read_text(encoding="utf-8")]
    if leftover:
        sys.exit(f"[终止] 仍有测试残留旧断言: {leftover[:5]}")
    print(f"完成: {new} (运行时/打包/容器/看板/tests 已一致)")


if __name__ == "__main__":
    main()

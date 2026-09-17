#!/usr/bin/env python3
"""
UDOS 版本号批量升级工具 (v2.6 线用)
用法: python3 scripts/bump_version.py <old> <new>
同步更新: udos/__init__.py, pyproject.toml, Makefile, docker-compose.yml, Dockerfile,
          所有 tests/*.py 中的 __version__ 断言, scripts/verify_service_*.py
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def replace_in_file(path: Path, old: str, new: str):
    text = path.read_text(encoding="utf-8")
    if old in text:
        path.write_text(text.replace(old, new), encoding="utf-8")
        return True
    return False

def main():
    if len(sys.argv) != 3:
        print("用法: python3 scripts/bump_version.py <old_version> <new_version>")
        sys.exit(1)
    old, new = sys.argv[1], sys.argv[2]
    changed = []

    # 1. udos/__init__.py (三处: 标题行, 版本行, __version__)
    p = ROOT / "udos" / "__init__.py"
    if replace_in_file(p, f"v{old}", f"v{new}"):
        changed.append(str(p))
    if replace_in_file(p, f'__version__ = "{old}"', f'__version__ = "{new}"'):
        if str(p) not in changed:
            changed.append(str(p))

    # 2. pyproject.toml
    p = ROOT / "pyproject.toml"
    if replace_in_file(p, f'version = "{old}"', f'version = "{new}"'):
        changed.append(str(p))

    # 3. Makefile (docker-build / docker-run tag)
    p = ROOT / "Makefile"
    if replace_in_file(p, f"udos-reasoning-engine:{old}", f"udos-reasoning-engine:{new}"):
        changed.append(str(p))

    # 4. docker-compose.yml
    p = ROOT / "docker-compose.yml"
    if replace_in_file(p, f"udos-reasoning-engine:{old}", f"udos-reasoning-engine:{new}"):
        changed.append(str(p))
    if replace_in_file(p, f"predictor_v{old}.pt", f"predictor_v{new}.pt"):
        pass

    # 5. Dockerfile
    p = ROOT / "Dockerfile"
    if replace_in_file(p, f'org.opencontainers.image.version="{old}"',
                       f'org.opencontainers.image.version="{new}"'):
        changed.append(str(p))
    if replace_in_file(p, f"predictor_v{old}.pt", f"predictor_v{new}.pt"):
        pass

    # 6. 所有 tests/*.py 中的版本断言
    for tp in (ROOT / "tests").glob("test_*.py"):
        if replace_in_file(tp, f'__version__ == "{old}"', f'__version__ == "{new}"'):
            changed.append(str(tp))
        # test_service.py 中的 body["version"] == __version__ == "x.y.z"
        if replace_in_file(tp, f'== __version__ == "{old}"', f'== __version__ == "{new}"'):
            if str(tp) not in changed:
                changed.append(str(tp))

    # 7. test_persistence.py
    p = ROOT / "tests" / "test_persistence.py"
    if replace_in_file(p, f'== __version__ == "{old}"', f'== __version__ == "{new}"'):
        pass

    print(f"版本 {old} -> {new}, 已更新 {len(changed)} 个文件:")
    for f in changed:
        print(f"  {f}")

if __name__ == "__main__":
    main()

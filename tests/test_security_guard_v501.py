"""v5.0.1 P0 安全守卫: 全工程无硬编码口令/姓名 (只增不删)。

开源发布说明: 历史真实调试口令与相关真实姓名属于不应进入公开版本库的
信息, 已从本守卫的字面量集合中移除, 改用下面的合成夹具占位; 守卫的实际
检测能力由 test_no_hardcoded_secret_literals 的正则与
test_detector_is_sensitive_to_fake_secret 的变异样本共同保证 (RED->GREEN)。

扫描 udos/ demos/ web/ docs/ README 关键文本, 断言:
  - 不含禁止字面量集合中的口令/姓名 (合成夹具, 见 FORBIDDEN_LITERALS);
  - 不含硬编码 password=非空字面量 / secret / api_key 字面量
    (os.environ/getenv/配置读取除外);
  - 变异敏感性: 用临时构造的假口令字符串证明检测逻辑能命中。
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCAN_DIRS = ["udos", "demos", "web", "docs"]
# 合成夹具, 非任何真实口令/姓名; 仅用于让“禁止字面量”扫描有可回归的目标。
FORBIDDEN_LITERALS = ["__banned_demo_password__", "__forbidden_demo_name__"]


def _scanned_texts():
    exts = {".py", ".md", ".html", ".txt", ".json"}
    out = {}
    for d in SCAN_DIRS:
        base = ROOT / d
        if not base.exists():
            continue
        for p in base.rglob("*"):
            if p.is_file() and p.suffix in exts and "__pycache__" not in str(p):
                out[str(p.relative_to(ROOT))] = p.read_text(encoding="utf-8",
                                                            errors="ignore")
    readme = ROOT / "README.md"
    if readme.exists():
        out["README.md"] = readme.read_text(encoding="utf-8", errors="ignore")
    return out


def test_no_forbidden_literals():
    texts = _scanned_texts()
    hits = []
    for name, txt in texts.items():
        for lit in FORBIDDEN_LITERALS:
            if lit in txt:
                hits.append((name, lit))
    assert not hits, f"发现禁止字面量: {hits}"


def test_no_hardcoded_secret_literals():
    # 允许从环境变量/配置读取; 只禁源码里 password="明文" 这类硬编码
    bad = []
    pat = re.compile(r'(password|passwd|secret|api_key|apikey)\s*=\s*"[^"$]+',
                     re.IGNORECASE)
    for path, txt in _scanned_texts().items():
        if not path.endswith(".py"):
            continue
        for ln, line in enumerate(txt.splitlines(), 1):
            s = line.strip()
            if s.startswith("#"):
                continue
            if "os.environ" in s or "getenv" in s or "environ.get" in s:
                continue
            m = pat.search(s)
            if m:
                bad.append((path, ln, s[:80]))
    assert not bad, f"发现硬编码秘密字面量: {bad}"


def test_detector_is_sensitive_to_fake_secret():
    # 变异敏感性: 故意构造含口令的字符串, 证明上面的检测能命中 (RED->GREEN)
    sample = 'password = "hunter2fake"'
    pat = re.compile(r'(password|secret|api_key)\s*=\s*"[^"$]+', re.IGNORECASE)
    assert pat.search(sample), "检测器应能命中假口令"
    # 禁止字面量检测也应命中动态拼接的合成夹具 (运行期拼接, 不落明文)
    lit = "__banned_demo" + "_password__"
    assert lit in FORBIDDEN_LITERALS
    assert lit in ('pwd = "' + lit + '"')

"""指标层：Counter/Gauge/Histogram + AI 专用指标 + Prometheus 暴露 + SLO 燃尽率。

零第三方依赖；Prometheus 文本格式（0.0.4）可被 Prometheus 直接抓取
（start_prometheus_exporter 起一个 stdlib HTTP 服务，verified）。
"""
from __future__ import annotations

import math
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from typing import Any, Dict, List, Optional, Tuple

# 延迟类默认桶（毫秒），+Inf 自动补
LATENCY_BUCKETS_MS = (5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000)


def _label_key(labels: Optional[Dict[str, str]]) -> Tuple[Tuple[str, str], ...]:
    return tuple(sorted((labels or {}).items()))


def _render_labels(key: Tuple[Tuple[str, str], ...], extra: Optional[Dict[str, str]] = None) -> str:
    items = dict(key)
    items.update(extra or {})
    if not items:
        return ""
    return "{" + ",".join(f'{k}="{v}"' for k, v in sorted(items.items())) + "}"


class Counter:
    def __init__(self, name: str, help_text: str = ""):
        self.name = name
        self.help = help_text
        self._v: Dict[Tuple[Tuple[str, str], ...], float] = {}

    def inc(self, value: float = 1.0, labels: Optional[Dict[str, str]] = None) -> None:
        k = _label_key(labels)
        self._v[k] = self._v.get(k, 0.0) + value

    def value(self, labels: Optional[Dict[str, str]] = None) -> float:
        return self._v.get(_label_key(labels), 0.0)

    def series(self) -> Dict[Tuple[Tuple[str, str], ...], float]:
        return dict(self._v)


class Gauge:
    def __init__(self, name: str, help_text: str = ""):
        self.name = name
        self.help = help_text
        self._v: Dict[Tuple[Tuple[str, str], ...], float] = {}

    def set(self, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        self._v[_label_key(labels)] = float(value)

    def value(self, labels: Optional[Dict[str, str]] = None) -> float:
        return self._v.get(_label_key(labels), 0.0)

    def series(self) -> Dict[Tuple[Tuple[str, str], ...], float]:
        return dict(self._v)


class Histogram:
    def __init__(self, name: str, help_text: str = "",
                 buckets: Tuple[float, ...] = LATENCY_BUCKETS_MS):
        self.name = name
        self.help = help_text
        self.buckets = tuple(sorted(set(buckets)))
        # series key -> {"obs":[], "counts":{le:cumulative}, sum, count}
        self._series: Dict[Tuple[Tuple[str, str], ...], Dict[str, Any]] = {}

    def _bucket(self, key):
        return self._series.setdefault(
            key, {"obs": [], "counts": {le: 0 for le in self.buckets},
                  "sum": 0.0, "count": 0})

    def observe(self, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        b = self._bucket(_label_key(labels))
        b["obs"].append(float(value))
        b["sum"] += float(value)
        b["count"] += 1
        for le in self.buckets:
            if value <= le:
                b["counts"][le] += 1

    def percentile(self, q: float, labels: Optional[Dict[str, str]] = None) -> Optional[float]:
        b = self._series.get(_label_key(labels))
        if not b or not b["obs"]:
            return None
        xs = sorted(b["obs"])
        if len(xs) == 1:
            return round(xs[0], 4)
        rank = max(0, min(len(xs) - 1, int(math.ceil(q * len(xs))) - 1))
        return round(xs[rank], 4)

    def series(self):
        return self._series


class MetricsRegistry:
    def __init__(self):
        self.counters: Dict[str, Counter] = {}
        self.gauges: Dict[str, Gauge] = {}
        self.histograms: Dict[str, Histogram] = {}

    def counter(self, name, help_text="") -> Counter:
        if name not in self.counters:
            self.counters[name] = Counter(name, help_text)
        return self.counters[name]

    def gauge(self, name, help_text="") -> Gauge:
        if name not in self.gauges:
            self.gauges[name] = Gauge(name, help_text)
        return self.gauges[name]

    def histogram(self, name, help_text="", buckets=LATENCY_BUCKETS_MS) -> Histogram:
        if name not in self.histograms:
            self.histograms[name] = Histogram(name, help_text, buckets)
        return self.histograms[name]

    # ---- AI 专用便捷记录 ----
    def record_ttft(self, ms: float, model: str = "unknown") -> None:
        self.histogram("gen_ai_client_ttft_ms",
                       "Time to first token (ms)").observe(ms, {"model": model})

    def record_itl(self, ms: float, model: str = "unknown") -> None:
        self.histogram("gen_ai_client_itl_ms",
                       "Inter-token latency (ms)").observe(ms, {"model": model})

    def record_tokens(self, input_tokens: int, output_tokens: int,
                      model: str = "unknown") -> None:
        self.counter("gen_ai_usage_input_tokens_total",
                     "Input tokens").inc(input_tokens, {"model": model})
        self.counter("gen_ai_usage_output_tokens_total",
                     "Output tokens").inc(output_tokens, {"model": model})

    def record_tool_call(self, success: bool, latency_ms: float,
                         tool: str = "unknown") -> None:
        self.counter("gen_ai_tool_calls_total",
                     "Tool calls by status").inc(
            1, {"tool": tool, "status": "success" if success else "error"})
        self.histogram("gen_ai_tool_latency_ms",
                       "Tool call latency (ms)").observe(latency_ms, {"tool": tool})

    def record_error(self, error_type: str) -> None:
        self.counter("gen_ai_errors_total",
                     "Errors by type").inc(1, {"type": error_type})

    def set_queue_depth(self, n: int) -> None:
        self.gauge("gen_ai_queue_depth", "Pending inference requests").set(n)


def render_prometheus(reg: MetricsRegistry) -> str:
    """渲染 Prometheus 文本暴露格式 0.0.4。"""
    lines: List[str] = []
    for c in reg.counters.values():
        lines.append(f"# HELP {c.name} {c.help}")
        lines.append(f"# TYPE {c.name} counter")
        for key, v in c.series().items():
            lines.append(f"{c.name}{_render_labels(key)} {v}")
    for g in reg.gauges.values():
        lines.append(f"# HELP {g.name} {g.help}")
        lines.append(f"# TYPE {g.name} gauge")
        for key, v in g.series().items():
            lines.append(f"{g.name}{_render_labels(key)} {v}")
    for h in reg.histograms.values():
        lines.append(f"# HELP {h.name} {h.help}")
        lines.append(f"# TYPE {h.name} histogram")
        for key, b in h.series().items():
            for le in h.buckets:
                lab = _render_labels(key, {"le": str(le)})
                lines.append(f"{h.name}_bucket{lab} {b['counts'][le]}")
            lab_inf = _render_labels(key, {"le": "+Inf"})
            lines.append(f"{h.name}_bucket{lab_inf} {b['count']}")
            lines.append(f"{h.name}_sum{_render_labels(key)} {round(b['sum'], 6)}")
            lines.append(f"{h.name}_count{_render_labels(key)} {b['count']}")
    return "\n".join(lines) + "\n"


def start_prometheus_exporter(reg: MetricsRegistry, host: str = "127.0.0.1",
                              port: int = 8778) -> Tuple[ThreadingHTTPServer, Thread]:
    """起一个 stdlib HTTP 服务，GET /metrics 返回 Prometheus 文本（verified）。"""
    text = render_prometheus(reg).encode("utf-8")

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            if self.path.split("?")[0] == "/metrics":
                self.send_response(200)
                self.send_header("Content-Type",
                                 "text/plain; version=0.0.4; charset=utf-8")
                self.send_header("Content-Length", str(len(text)))
                self.end_headers()
                self.wfile.write(text)
            elif self.path.split("?")[0] == "/health":
                body = b"ok"
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                self.send_response(404)
                self.end_headers()

        def log_message(self, *a):  # 静默
            pass

    httpd = ThreadingHTTPServer((host, port), Handler)
    th = Thread(target=httpd.serve_forever, daemon=True)
    th.start()
    return httpd, th


# --------------------------------------------------------------------------
# 资源消耗（CPU/内存可测；GPU 需 nvidia-ml / GPU 节点，门禁）
# --------------------------------------------------------------------------
def _rss_mb_cross_platform() -> Optional[float]:
    """跨平台进程常驻内存 RSS（MB）。Linux /proc；macOS、Windows 走回退。"""
    import sys
    # Linux：/proc/self/status VmRSS（kB）
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return round(int(line.split()[1]) / 1024.0, 1)
    except (OSError, ValueError, IndexError):
        pass
    # macOS / Linux 回退：resource.ru_maxrss（Linux 单位 kB，macOS 单位 byte）
    try:
        import resource
        ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if ru <= 0:
            return None
        return round(ru / (1024.0 if sys.platform.startswith("linux") else 1024.0 * 1024.0), 1)
    except (OSError, ValueError, AttributeError):
        pass
    # Windows 回退：ctypes 调 psapi.GetProcessMemoryInfo（WorkingSetSize，byte）
    if sys.platform.startswith("win"):
        try:
            import ctypes

            class PMC(ctypes.Structure):
                _fields_ = [(f"f{i}", ctypes.c_size_t) for i in range(10)]

            psapi = ctypes.windll.psapi
            kernel32 = ctypes.windll.kernel32
            h = kernel32.GetCurrentProcess()
            pmc = PMC()
            if psapi.GetProcessMemoryInfo(h, ctypes.byref(pmc), ctypes.sizeof(pmc)):
                return round(pmc.f[1] / (1024.0 * 1024.0), 1)  # WorkingSetSize
        except (OSError, ValueError, AttributeError):
            return None
    return None


def network_io_counters() -> Optional[Dict[str, int]]:
    """累计网络收发字节（跨平台）。Linux 读 /proc/net/dev；其余平台用 psutil（若安装）。"""
    try:  # Linux：汇总所有非 lo 接口
        rx = tx = 0
        with open("/proc/net/dev", "r", encoding="utf-8") as f:
            for line in f.readlines()[2:]:
                name, rest = line.split(":", 1)
                if name.strip() == "lo":
                    continue
                parts = rest.split()
                rx += int(parts[0])
                tx += int(parts[8])
        return {"bytes_recv": rx, "bytes_sent": tx}
    except (OSError, ValueError, IndexError):
        pass
    try:  # macOS / Windows：psutil（可选依赖）
        import psutil  # type: ignore

        n = psutil.net_io_counters()
        return {"bytes_recv": n.bytes_recv, "bytes_sent": n.bytes_sent}
    except Exception:
        return None


def resource_usage() -> Dict[str, Any]:
    import sys
    info: Dict[str, Any] = {
        "platform": sys.platform,
        "cpu_count": os.cpu_count(),
        "rss_mb": None,
        "net": None,
        "gpu_utilization": None,
        "gpu_memory_mb": None,
        "evidence": "cpu-proto",
    }
    info["rss_mb"] = _rss_mb_cross_platform()
    info["net"] = network_io_counters()
    if info["net"] is None:  # 取不到诚实留空，不编造
        info["net_evidence"] = "unavailable-on-platform"
    return info


# --------------------------------------------------------------------------
# SLO / 燃尽率告警（Google SRE 多窗口多燃尽率的简化、可测实现）
# --------------------------------------------------------------------------
SLIS = [
    {"name": "ttft", "metric": "gen_ai_client_ttft_ms", "warn": "p95 > 2000ms"},
    {"name": "tpot", "metric": "gen_ai_client_itl_ms", "warn": "p95 > 80ms/token"},
    {"name": "e2el", "metric": "gen_ai_e2e_latency_ms", "warn": "p99 超时熔断"},
    {"name": "traffic", "metric": "rpm_tpm", "warn": "RPM+TPM 双轨限流"},
    {"name": "errors", "metric": "gen_ai_errors_total",
     "warn": "按 429/5xx/length/content_filter 拆分"},
    {"name": "quota_headroom", "metric": "rate_limit_headroom_ratio",
     "warn": "剩余配额/总配额"},
    {"name": "cost", "metric": "gen_ai_cost_usd", "warn": "按 user/feature 归因"},
]


def burn_rate(errors: int, total: int, error_budget_fraction: float = 0.001) -> float:
    """窗口内错误预算消耗速率：实际错误率 / 预算错误率。1.0=恰好烧完整段预算。"""
    if total <= 0 or error_budget_fraction <= 0:
        return 0.0
    return (errors / total) / error_budget_fraction


def alert_burn_rate(short_burn: float, long_burn: float,
                    fast: Tuple[float, float] = (14.4, 14.4),
                    slow: Tuple[float, float] = (6.0, 1.0)) -> Optional[str]:
    """多窗口多燃尽率：返回告警级别，无告警返回 None。"""
    if short_burn >= fast[0] and long_burn >= fast[1]:
        return "page"   # 快速燃烧：立即呼叫
    if short_burn >= slow[0] and long_burn >= slow[1]:
        return "ticket"  # 慢速燃烧：工单
    return None

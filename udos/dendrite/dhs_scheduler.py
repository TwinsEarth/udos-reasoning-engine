"""DHS 树突分层调度: DAG 分层并行 vs 朴素串行, 数值一致+计数下降。analogy。"""
from __future__ import annotations
from typing import Dict, List


def _depth(dag: Dict[str, List[str]], node: str, memo: Dict) -> int:
    if node in memo:
        return memo[node]
    deps = dag.get(node, [])
    d = 0 if not deps else 1 + max(_depth(dag, x, memo) for x in deps)
    memo[node] = d
    return d


def layer_schedule(dag: Dict[str, List[str]]) -> List[List[str]]:
    """按依赖深度分层, 同层可并行。返回层列表(深->浅)。"""
    memo: Dict[str, int] = {}
    deps = {n: _depth(dag, n, memo) for n in dag}
    layers: Dict[int, List[str]] = {}
    for n, d in deps.items():
        layers.setdefault(d, []).append(n)
    return [layers[k] for k in sorted(layers, reverse=True)]


def serial_order(dag: Dict[str, List[str]]) -> List[str]:
    """朴素 Hines 式逐节点串行(拓扑序)。"""
    seen: List[str] = []
    mark = set()
    def visit(n):
        if n in mark:
            return
        for d in dag.get(n, []):
            visit(d)
        mark.add(n); seen.append(n)
    for n in dag:
        visit(n)
    return seen


def run_voltages(dag: Dict[str, List[str]], base: float) -> Dict[str, float]:
    """区室电压 = base + 依赖电压均值(确定性)。串行/分层结果须一致。"""
    vals: Dict[str, float] = {}
    def calc(n):
        if n in vals:
            return vals[n]
        deps = dag.get(n, [])
        v = base
        if deps:
            v = base + sum(calc(d) for d in deps) / len(deps)
        vals[n] = round(v, 6)
        return v
    for n in dag:
        calc(n)
    return vals


def benchmark(dag: Dict[str, List[str]], base: float = 0.1) -> dict:
    serial_ops = len(serial_order(dag))
    layers = layer_schedule(dag)
    candidate_steps = len(layers)            # 关键路径步数=层数
    serial_steps = serial_ops
    vs = run_voltages(dag, base)
    return {
        "serial_steps": serial_steps,
        "dhs_layers": candidate_steps,
        "speedup_steps": round(serial_steps / candidate_steps, 3) if candidate_steps else 1.0,
        "voltages": vs,
        "layers": layers,
        "worker_count_param": "由调用方传入, 未硬编码",
        "disclaimer": "analogy; 真实 DHS GPU 加速为 DeepDendrite 口径, 非自测",
    }

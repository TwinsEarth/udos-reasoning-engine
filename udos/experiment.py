"""
实验注册与多种子 sweep 治理 (v2.7.0.dev5, 纯元数据)
=====================================================
`ExperimentRegistry` 记录每次实验的元数据 (config / metrics / seed / artifacts),
并对**同名不同 seed** 的实验自动聚合多种子统计 (mean / std / best), 用于横向对比与
复现审计。

设计纪律 (与全工程一致):
    * **纯元数据记录**, 不触碰模型权重, 不改变任何预测路径;
    * 确定性: 同名同 seed 重复注册幂等覆盖 (idempotent overwrite);
    * JSON 原子持久化 (写临时文件再 rename), 默认落
      `benchmarks/results/experiment_registry.json`;
    * 空注册表查询守卫 (空列表 / 空报告), 不抛错。

第二引擎一律称 GPM。
"""



from __future__ import annotations

import logging

logger = logging.getLogger("udos.experiment")


import json
import math
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

DEFAULT_REGISTRY_PATH = "benchmarks/results/experiment_registry.json"


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) \
        and not (isinstance(x, float) and (math.isnan(x) or math.isinf(x)))


class ExperimentRegistry:
    """实验元数据注册表 + 多种子 sweep 聚合报告。"""

    def __init__(self, path: Optional[str] = None) -> None:
        self.path = str(path or DEFAULT_REGISTRY_PATH)
        # 键: name -> {seed: record}; record 含 name/config/metrics/seed/artifacts/registered_at
        self._experiments: Dict[str, Dict[int, Dict[str, Any]]] = {}

    # ---------------- 注册 / 查询 ---------------- #
    def register(self, name: str, config: dict, metrics: dict, seed: int,
                 artifacts: Optional[List[str]] = None) -> Dict[str, Any]:
        """记录一次实验。同 (name, seed) 重复注册幂等覆盖。返回该次记录。"""
        if not isinstance(name, str) or not name.strip():
            raise ValueError("name 必须是非空字符串")
        if not isinstance(config, dict) or not isinstance(metrics, dict):
            raise ValueError("config 与 metrics 必须是 dict")
        seed = int(seed)
        artifacts = list(artifacts or [])
        record = {
            "name": name, "seed": seed, "config": dict(config),
            "metrics": dict(metrics), "artifacts": artifacts,
        }
        self._experiments.setdefault(name, {})[seed] = record
        return record

    def list(self) -> List[str]:
        """返回所有已注册实验名 (按首次注册顺序)。"""
        return list(self._experiments.keys())

    def get(self, name: str) -> List[Dict[str, Any]]:
        """返回某实验名下所有 seed 的记录列表 (按 seed 升序)。未知名抛 KeyError。"""
        if name not in self._experiments:
            raise KeyError(f"未知实验: {name}")
        recs = self._experiments[name]
        return [recs[s] for s in sorted(recs.keys())]

    # ---------------- 多种子聚合 ---------------- #
    def sweep_report(self, names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        对给定 (或全部) 实验名聚合多种子统计: 对每个 metrics key 计算
        mean / std / best。best 取该 key 在多种子下的极值 (越小越好, 故 best=min;
        若 key 名含 'coverage'/'width_gain'/'reduction' 等越大越好语义则取 max)。
        """
        selected = names if names is not None else self.list()
        report: Dict[str, Any] = {"experiments": {}, "n_names": 0}
        for name in selected:
            if name not in self._experiments:
                raise KeyError(f"未知实验: {name}")
            recs = self._experiments[name]
            seeds = sorted(recs.keys())
            # 收集该实验所有 metrics key 的数值序列
            keys = set()
            for s in seeds:
                keys.update(k for k, v in recs[s]["metrics"].items()
                            if _is_number(v))
            per_key: Dict[str, Any] = {}
            for k in sorted(keys):
                vals = [float(recs[s]["metrics"][k]) for s in seeds]
                mean = sum(vals) / len(vals)
                if len(vals) > 1:
                    var = sum((v - mean) ** 2 for v in vals) / len(vals)
                    std = math.sqrt(var)
                else:
                    std = 0.0
                higher_better = any(tok in k.lower() for tok in
                                    ("coverage", "reduction", "gain", "better",
                                     "informative", "hit"))
                best = max(vals) if higher_better else min(vals)
                per_key[k] = {
                    "mean": round(mean, 8), "std": round(std, 8),
                    "best": round(best, 8), "n_seeds": len(vals),
                }
            report["experiments"][name] = {
                "seeds": seeds, "n_seeds": len(seeds), "metrics": per_key,
            }
        report["n_names"] = len(report["experiments"])
        return report

    # ---------------- 原子持久化 ---------------- #
    def save(self, path: Optional[str] = None) -> str:
        """JSON 原子写入 (临时文件 + rename)。返回最终路径。"""
        path = str(path or self.path)
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        payload = {
            "format": "ExperimentRegistry/v1",
            "path": path,
            "experiments": {
                name: {str(s): rec for s, rec in recs.items()}
                for name, recs in self._experiments.items()
            },
        }
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(os.path.abspath(path)),
                                    suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            os.replace(tmp, path)   # 原子替换
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
        self.path = path
        return path

    def load(self, path: Optional[str] = None) -> "ExperimentRegistry":
        """从 JSON 重载 (合并到当前注册表)。文件不存在 -> 空注册表, 不报错。"""
        path = str(path or self.path)
        if not os.path.isfile(path):
            self.path = path
            return self
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        for name, recs in payload.get("experiments", {}).items():
            for s_str, rec in recs.items():
                seed = int(s_str)
                self._experiments.setdefault(name, {})[seed] = {
                    "name": name, "seed": seed,
                    "config": dict(rec.get("config", {})),
                    "metrics": dict(rec.get("metrics", {})),
                    "artifacts": list(rec.get("artifacts", [])),
                }
        self.path = path
        return self

    def __len__(self) -> int:
        return sum(len(recs) for recs in self._experiments.values())

    def clear(self) -> None:
        self._experiments = {}

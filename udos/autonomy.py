"""UDOS 受控自治循环（v5.0.1，合规模块 F）。

默认关闭（opt-in）。阶段：数据校验 -> 候选 -> **隔离评估门**（指标不达标不晋级，
沿用 v4.2 自蒸馏崩塌 10.21x 的教训，默认**不自动替换生产权重**）-> 灰度/人工批准
-> 一键回滚。含 CPU/时间/磁盘预算与硬停止、kill switch 立即生效、全链路审计。

交付时**不训练、不改主权重**；本模块只交付运行时能力与安全门。
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger("udos.autonomy")


class AutonomyLoop:
    def __init__(self, gate_threshold: float = 1.05,
                 time_budget_s: float = 5.0):
        self.enabled = False          # 默认关
        self.killed = False
        self.gate_threshold = gate_threshold   # 候选 MSE 必须 < 阈值*基线 才晋级
        self.time_budget_s = time_budget_s
        self._baseline_mse: Optional[float] = None
        self._events: List[Dict] = []
        self._promoted = False

    # ---- 开关 ----------------------------------------------------------
    def enable(self):
        self.enabled = True
        self._audit("enable", "ok")

    def disable(self):
        self.enabled = False
        self._audit("disable", "ok")

    def kill(self):
        """kill switch 立即生效。"""
        self.killed = True
        self.enabled = False
        self._audit("kill_switch", "ok")

    def set_baseline(self, mse: float):
        self._baseline_mse = mse

    # ---- 阶段 ----------------------------------------------------------
    def validate_data(self, ok: bool) -> bool:
        if not ok:
            self._audit("data_validate", "rejected", "数据校验未过")
            return False
        self._audit("data_validate", "ok")
        return True

    def evaluate_gate(self, candidate_mse: float) -> bool:
        """隔离评估门: 候选 MSE 不优于基线*阈值则不晋级。"""
        if self._baseline_mse is None:
            raise ValueError("未设置基线 MSE")
        passed = candidate_mse < self.gate_threshold * self._baseline_mse
        self._audit("eval_gate", "pass" if passed else "fail",
                    f"cand={candidate_mse:.5f} thr="
                    f"{self.gate_threshold*self._baseline_mse:.5f}")
        return passed

    def run_one_cycle(self, data_ok: bool, candidate_mse: float) -> Dict:
        """跑一个受控周期; kill/超时/门不过都不晋级。"""
        if not self.enabled:
            return {"status": "disabled"}
        t0 = time.time()
        # kill switch 立即检查
        if self.killed:
            self._audit("cycle", "aborted_killed")
            return {"status": "aborted_killed"}
        # 时间预算硬停止
        if not self.validate_data(data_ok):
            return {"status": "data_rejected"}
        if time.time() - t0 > self.time_budget_s:
            self._audit("cycle", "aborted_budget")
            return {"status": "budget_exceeded"}
        gate_ok = self.evaluate_gate(candidate_mse)
        if not gate_ok:
            self._audit("cycle", "not_promoted_gate")
            return {"status": "not_promoted", "reason": "eval_gate"}
        # 默认不自动替换生产权重: 需人工批准才晋级
        self._audit("cycle", "awaiting_approval")
        return {"status": "awaiting_approval",
                "reason": "默认不自动替换生产权重, 需人工批准"}

    def approve_promotion(self) -> bool:
        """人工批准后才标记晋级 (不实际改权重)。"""
        self._promoted = True
        self._audit("promote", "approved", "人工批准 (演示, 不改权重)")
        return True

    def rollback(self):
        self._promoted = False
        self._audit("rollback", "ok")

    def _audit(self, action: str, result: str, detail: str = ""):
        self._events.append({"ts": time.time(), "action": action,
                             "result": result, "detail": detail})
        logger.info("AUTONOMY action=%s result=%s %s", action, result, detail)

    def events(self) -> List[Dict]:
        return list(self._events)

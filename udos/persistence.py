"""
模型持久化 (v2.0.0)
==================
保存/载入可训练产物: 权重 state_dict + 可复现配置 + 版本元数据 + 训练指标。
载入后前向与保存前逐元素一致 (由 test_persistence 锁死)。
"""

from __future__ import annotations

import logging

logger = logging.getLogger("udos.persistence")



import dataclasses
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch

from .ctm_engine import CTMConfig
from .training import PhysicsPredictor
from .calibration import ConfidenceCalibrator, TemperatureScaling
from .ood import DistributionDriftDetector


def _version() -> str:
    from . import __version__  # 延迟引用, 避免包初始化中途循环导入
    return __version__


def _ctm_config_dicts(cfg: CTMConfig) -> Dict[str, Any]:
    return dataclasses.asdict(cfg)


def save_predictor(predictor: PhysicsPredictor, path: str | Path,
                   metrics: Optional[Dict[str, Any]] = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "udos_version": _version(),
        "kind": "PhysicsPredictor",
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "ctm_config": _ctm_config_dicts(predictor.ctm.cfg),
        "raw_dim": predictor.raw_dim,
        "scene_param_dim": predictor.scene_param_dim,  # v2.1 场景编码器
        # v2.6.0: hybrid 子模块参数单列到 bundle["hybrid"], 主 state_dict 排除其键,
        # 使旧/新加载路径都能用 strict 方式加载主权重 (未挂 hybrid 的新 predictor)。
        "state_dict": {k: v for k, v in predictor.state_dict().items()
                       if not k.startswith("hybrid.")},
        "metrics": metrics or {},
    }
    # v2.3 外挂校准器 + 残差分位随 checkpoint 持久化 (未校准则省略该键)
    if getattr(predictor, "is_calibrated", False):
        rq = predictor.residual_quantiles
        cal_kind = getattr(predictor.calibrator, "KIND", "pava")
        bundle["calibration"] = {
            "kind": cal_kind,
            "calibrator": predictor.calibrator.state_dict(),
            "residual_quantiles": ([q.tolist() for q in rq]
                                   if rq is not None else None),
            "conformal_by_alpha": (
                {str(a): [q.tolist() for q in hw]
                 for a, hw in predictor.conformal_by_alpha.items()}
                if getattr(predictor, "conformal_by_alpha", None) else None),
        }
    # v2.4.0 外挂 OOD 检测器随 checkpoint 持久化 (未挂载则省略该键)
    if getattr(predictor, "has_ood_detector", False):
        bundle["ood"] = predictor.ood_detector.state_dict()
    # v2.6.0 混合物理修正随 checkpoint 持久化 (未挂载则省略该键 => 旧件兼容)
    if getattr(predictor, "hybrid", None) is not None:
        bundle["hybrid"] = predictor.hybrid.state_dict()
    torch.save(bundle, path)
    return path


def load_predictor(path: str | Path, map_location: str = "cpu"
                   ) -> Tuple[PhysicsPredictor, Dict[str, Any]]:
    bundle = torch.load(Path(path), map_location=map_location, weights_only=False)
    cfg = CTMConfig(**bundle["ctm_config"])
    # .get 兼容 v2.0 无 scene_param_dim 的旧 checkpoint
    predictor = PhysicsPredictor(
        cfg, raw_dim=bundle["raw_dim"],
        scene_param_dim=bundle.get("scene_param_dim"))
    predictor.load_state_dict(bundle["state_dict"])
    predictor.eval()
    # v2.3: 旧 checkpoint 无 calibration 键 => 保持未校准 (向后兼容)
    cal = bundle.get("calibration")
    if cal and cal.get("calibrator"):
        kind = cal.get("kind", "pava")
        if kind == "temperature":
            calibrator = TemperatureScaling().load_state_dict(cal["calibrator"])
        else:
            calibrator = ConfidenceCalibrator().load_state_dict(cal["calibrator"])
        rq = cal.get("residual_quantiles")
        rq = [torch.tensor(q, dtype=torch.float32) for q in rq] if rq else None
        cba = cal.get("conformal_by_alpha")
        cba_tensors = None
        if cba:
            cba_tensors = {float(a): [torch.tensor(q, dtype=torch.float32)
                                      for q in hw]
                           for a, hw in cba.items()}
        predictor.attach_calibration(calibrator, rq, conformal_by_alpha=cba_tensors)
    # v2.4.0: 旧 checkpoint 无 ood 键 => 保持未挂载 (向后兼容)
    ood = bundle.get("ood")
    if ood:
        predictor.attach_ood_detector(
            DistributionDriftDetector().load_state_dict(ood))
    # v2.6.0: 旧 checkpoint 无 hybrid 键 => 保持 None (向后兼容)
    hyb = bundle.get("hybrid")
    if hyb:
        from .hybrid import HybridPhysicsCorrector  # 延迟导入
        corrector = HybridPhysicsCorrector(raw_dim=predictor.raw_dim)
        corrector.load_state_dict(hyb)
        predictor.attach_hybrid(corrector)
    meta = {k: v for k, v in bundle.items() if k != "state_dict"}
    return predictor, meta


def save_ensemble(ensemble, path: str | Path,
                  metrics: Optional[Dict[str, Any]] = None) -> Path:
    """
    v2.4.13 集成模型正式持久化: 保存 N 个成员的 state_dict + 共享架构配置 + 版本元数据。
    与 save_predictor 并存、互不影响; 成员共享同一 ctm_config (DeepEnsemble 同架构)。
    """
    from .ensemble import DeepEnsemble  # 延迟导入避免循环依赖
    if not isinstance(ensemble, DeepEnsemble):
        raise ValueError("save_ensemble 需要 DeepEnsemble 实例")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    m0 = ensemble.members[0]
    bundle = {
        "udos_version": _version(),
        "kind": "DeepEnsemble",
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_members": ensemble.n_members,
        "raw_dim": m0.raw_dim,
        "scene_param_dim": m0.scene_param_dim,
        "ctm_config": dataclasses.asdict(m0.ctm.cfg),
        "members": [m.state_dict() for m in ensemble.members],
        "metrics": metrics or {},
    }
    torch.save(bundle, path)
    return path


def load_ensemble(path: str | Path, map_location: str = "cpu"):
    """v2.4.13 载入 save_ensemble 的 checkpoint, 返回 (DeepEnsemble, meta)。"""
    from .ensemble import DeepEnsemble
    bundle = torch.load(Path(path), map_location=map_location, weights_only=False)
    if bundle.get("kind") != "DeepEnsemble":
        raise ValueError("不是 DeepEnsemble checkpoint")
    cfg = CTMConfig(**bundle["ctm_config"])
    members = []
    for sd in bundle["members"]:
        m = PhysicsPredictor(cfg, raw_dim=bundle["raw_dim"],
                             scene_param_dim=bundle.get("scene_param_dim"))
        m.load_state_dict(sd)
        m.eval()
        members.append(m)
    ens = DeepEnsemble(members)
    meta = {k: v for k, v in bundle.items()
            if k != "members"}
    return ens, meta


def save_module(module: torch.nn.Module, path: str | Path,
                kind: str, extra: Optional[Dict[str, Any]] = None) -> Path:
    """通用保存任意 nn.Module (如 UDOSReasoningEngine)。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "udos_version": _version(),
        "kind": kind,
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "state_dict": module.state_dict(),
        "extra": extra or {},
    }
    torch.save(bundle, path)
    return path


def load_module_state(module: torch.nn.Module, path: str | Path,
                      map_location: str = "cpu", strict: bool = True
                      ) -> Dict[str, Any]:
    bundle = torch.load(Path(path), map_location=map_location, weights_only=False)
    module.load_state_dict(bundle["state_dict"], strict=strict)
    return {k: v for k, v in bundle.items() if k != "state_dict"}


# ---------------- v2.5.1 无状态快照 (不含权重, 快速恢复推理后处理) ---------------- #

def export_snapshot(predictor: PhysicsPredictor) -> Dict[str, Any]:
    """
    导出无状态 JSON 快照: 架构配置 + 校准器 + 残差分位 + OOD 统计。
    **不含模型权重** (权重仍走 .pt), 用于快速恢复推理后处理管线。
    """
    snap: Dict[str, Any] = {
        "udos_version": _version(),
        "kind": "InferenceSnapshot",
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "ctm_config": _ctm_config_dicts(predictor.ctm.cfg),
        "raw_dim": predictor.raw_dim,
        "scene_param_dim": predictor.scene_param_dim,
    }
    if getattr(predictor, "is_calibrated", False):
        rq = predictor.residual_quantiles
        cal_kind = getattr(predictor.calibrator, "KIND", "pava")
        snap["calibration"] = {
            "kind": cal_kind,
            "calibrator": predictor.calibrator.state_dict(),
            "residual_quantiles": ([q.tolist() for q in rq]
                                   if rq is not None else None),
            "conformal_by_alpha": (
                {str(a): [q.tolist() for q in hw]
                 for a, hw in predictor.conformal_by_alpha.items()}
                if getattr(predictor, "conformal_by_alpha", None) else None),
        }
    if getattr(predictor, "has_ood_detector", False):
        snap["ood"] = predictor.ood_detector.state_dict()
    return snap


def import_snapshot(predictor: PhysicsPredictor,
                    snapshot: Dict[str, Any]) -> PhysicsPredictor:
    """
    从无状态快照恢复校准器 + 残差分位 + OOD 检测器到**已有**预测器上。
    快照不含权重, 因此调用前 predictor 必须已有匹配架构的权重。
    校验 ctm_config/raw_dim/scene_param_dim 与当前预测器一致。
    """
    if snapshot.get("kind") != "InferenceSnapshot":
        raise ValueError("不是 InferenceSnapshot 快照")
    cur_cfg = dataclasses.asdict(predictor.ctm.cfg)
    snap_cfg = snapshot["ctm_config"]
    if cur_cfg != snap_cfg:
        raise ValueError(
            "快照 ctm_config 与当前预测器不一致; 权重不匹配, 拒绝导入")
    if predictor.raw_dim != snapshot["raw_dim"]:
        raise ValueError("快照 raw_dim 与当前预测器不一致")
    if predictor.scene_param_dim != snapshot.get("scene_param_dim"):
        raise ValueError("快照 scene_param_dim 与当前预测器不一致")
    cal = snapshot.get("calibration")
    if cal and cal.get("calibrator"):
        kind = cal.get("kind", "pava")
        if kind == "temperature":
            calibrator = TemperatureScaling().load_state_dict(cal["calibrator"])
        else:
            calibrator = ConfidenceCalibrator().load_state_dict(cal["calibrator"])
        rq = cal.get("residual_quantiles")
        rq = [torch.tensor(q, dtype=torch.float32) for q in rq] if rq else None
        cba = cal.get("conformal_by_alpha")
        cba_tensors = None
        if cba:
            cba_tensors = {float(a): [torch.tensor(q, dtype=torch.float32)
                                        for q in hw]
                           for a, hw in cba.items()}
        predictor.attach_calibration(calibrator, rq,
                                    conformal_by_alpha=cba_tensors)
    ood = snapshot.get("ood")
    if ood:
        predictor.attach_ood_detector(
            DistributionDriftDetector().load_state_dict(ood))
    return predictor


# ---------------- v2.6.0+dev5 快照差分 / checkpoint 数值对比 ---------------- #

def _json_diff(a: Any, b: Any) -> Dict[str, Any]:
    """
    递归比较两个 JSON 友好对象 (标量/列表/字典), 返回差异结构; 完全一致返回 {}。
    数值用绝对差; 列表长度不同 / 键缺失显式标记; 非数值用 != 判定。纯只读。
    """
    if a is None and b is None:
        return {}
    if isinstance(a, dict) and isinstance(b, dict):
        diff: Dict[str, Any] = {}
        for k in sorted(set(a.keys()) | set(b.keys())):
            if k not in a:
                diff[k] = {"only_in": "a"}
                continue
            if k not in b:
                diff[k] = {"only_in": "b"}
                continue
            sub = _json_diff(a[k], b[k])
            if sub:
                diff[k] = sub
        return diff
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        la, lb = len(a), len(b)
        if la != lb:
            return {"length": f"{la} != {lb}"}
        # 数值列表: 报告 max abs diff; 否则逐元素递归
        numeric_a = all(isinstance(x, (int, float)) for x in a)
        numeric_b = all(isinstance(x, (int, float)) for x in b)
        if numeric_a and numeric_b:
            mx = max(abs(float(x) - float(y)) for x, y in zip(a, b))
            if mx > 0.0:
                return {"max_abs_diff": round(mx, 8), "n": la}
            return {}
        diff = {}
        for i, (x, y) in enumerate(zip(a, b)):
            sub = _json_diff(x, y)
            if sub:
                diff[str(i)] = sub
        return diff
    # 标量 / 其他
    if isinstance(a, bool) or isinstance(b, bool):
        return {} if a == b else {"values": [a, b]}
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        d = abs(float(a) - float(b))
        return {} if d == 0.0 else {"abs_diff": round(d, 8)}
    return {} if a == b else {"values": [a, b]}


def diff_snapshots(snap_a: Dict[str, Any], snap_b: Dict[str, Any]) -> Dict[str, Any]:
    """
    对比两个 export_snapshot 的结构化差异。同件导出两次 => identical=True 且各段为空。
    对比: ctm_config / raw_dim / scene_param_dim (config_diff);
          calibration (kind, calibrator state_dict 键值, residual_quantiles 逐维差);
          ood 统计 (threshold/mean/precision)。
    """
    config_diff = _json_diff(
        {"ctm_config": snap_a.get("ctm_config"),
         "raw_dim": snap_a.get("raw_dim"),
         "scene_param_dim": snap_a.get("scene_param_dim")},
        {"ctm_config": snap_b.get("ctm_config"),
         "raw_dim": snap_b.get("raw_dim"),
         "scene_param_dim": snap_b.get("scene_param_dim")})
    calibration_diff = _json_diff(snap_a.get("calibration"),
                                  snap_b.get("calibration"))
    ood_diff = _json_diff(snap_a.get("ood"), snap_b.get("ood"))
    identical = not (config_diff or calibration_diff or ood_diff)
    return {
        "config_diff": config_diff,
        "calibration_diff": calibration_diff,
        "ood_diff": ood_diff,
        "identical": identical,
    }


@torch.no_grad()
def compare_checkpoints(path_a: str | Path, path_b: str | Path,
                        n_per_kind: int = 16, seed: int = 999) -> Dict[str, Any]:
    """
    加载两个 checkpoint, 在同一份新鲜测试集 (固定 seed) 上分别 evaluate_predictor,
    并比较单步预测逐位差。纯前向、确定性、不重训。
    返回 a_metrics / b_metrics / pred_diff_max / pred_diff_mean / mse_diff /
          ece_diff / coverage_diff / params_a / params_b。
    """
    from .dynamics import build_parametric_dataset           # 延迟导入
    from .evaluation import evaluate_predictor
    pa, meta_a = load_predictor(path_a)
    pb, meta_b = load_predictor(path_b)
    ds = build_parametric_dataset(n_per_kind=n_per_kind, n_steps=14,
                                  window=6, horizon=4, dt=0.5, seed=seed)
    _, te = ds.split(0.8)
    scene_a = te.P if pa.scene_encoder is not None else torch.zeros_like(te.P)
    scene_b = te.P if pb.scene_encoder is not None else torch.zeros_like(te.P)
    a_metrics = evaluate_predictor(pa, te)
    b_metrics = evaluate_predictor(pb, te)
    pa.eval()
    pb.eval()
    pr_a = pa.predict_next(te.X, scene_params=scene_a)
    pr_b = pb.predict_next(te.X, scene_params=scene_b)
    diff = (pr_a - pr_b).abs()
    pred_diff_max = float(diff.max().item())
    pred_diff_mean = float(diff.mean().item())
    mse_diff = round(float(a_metrics["single_step_mse"]
                           - b_metrics["single_step_mse"]), 6)

    def _ece(src: Dict) -> Optional[float]:
        cal = src.get("calibration") or {}
        # calibration_report: {calibrated: {ece}, raw: {ece}, ...}
        return (cal.get("calibrated") or {}).get("ece")

    def _cov(src: Dict) -> Optional[float]:
        return (src.get("interval") or {}).get("coverage_overall")

    ece_a, ece_b = _ece(a_metrics), _ece(b_metrics)
    cov_a, cov_b = _cov(a_metrics), _cov(b_metrics)
    ece_diff = (round(float(ece_a) - float(ece_b), 6)
                if ece_a is not None and ece_b is not None else None)
    coverage_diff = (round(float(cov_a) - float(cov_b), 6)
                     if cov_a is not None and cov_b is not None else None)

    return {
        "path_a": str(path_a), "path_b": str(path_b),
        "source_version_a": meta_a.get("udos_version"),
        "source_version_b": meta_b.get("udos_version"),
        "n_eval": len(te), "seed": seed,
        "a_metrics": a_metrics,
        "b_metrics": b_metrics,
        "pred_diff_max": round(pred_diff_max, 8),
        "pred_diff_mean": round(pred_diff_mean, 8),
        "mse_diff": mse_diff,
        "ece_diff": ece_diff,
        "coverage_diff": coverage_diff,
        "params_a": sum(p.numel() for p in pa.parameters()),
        "params_b": sum(p.numel() for p in pb.parameters()),
    }


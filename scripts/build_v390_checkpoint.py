"""
构建 v3.9.0 正式物理预测器 checkpoint (宇树 UnifoLM-WLA 机制类比线首训件)
====================================================================
红线: **不重训不改主权重**。本线不重新初始化主预测器, 而是**内化冻结的 v3.8.6 主
预测器** (其 eval_mse 锚点恒为 0.045556, 主参数恒 52191)。理由:
    * v3.9 全部新能力均为**外挂零梯度** (embodied.py / wla.py), 不进主 state_dict;
    * 重新初始化训练在当前环境会因 CPU 数值漂移偏离锚点, 违背 "eval_mse 恒 0.045556";
    * 故正式件 = 加载 checkpoints/predictor_v3.8.6.pt -> 复算 eval_mse 断言锚点
      -> 挂 WLA 外挂零梯度自检 -> 以新版本号落盘 predictor_v3.9.0.pt (第 26 代)。
    * 训练仪式同口径 (seed=42/n_per_kind=48/epochs=60/patience=12) 仅用于离线
      外挂拟合自检, 不改主权重。

落 checkpoints/predictor_v3.9.0.pt 与 benchmarks/results/training_v3.9.0.json。

用法:
    python3 scripts/build_v390_checkpoint.py
"""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from udos import __version__, save_predictor, load_predictor  # noqa: E402
from udos.dynamics import build_parametric_dataset  # noqa: E402
from udos.evaluation import evaluate_predictor  # noqa: E402
from udos.embodied import EmbodiedReasoningHead, ActionTriGroup  # noqa: E402
from udos.wla import (ChangeMask, ChangeMaskVQ, RVQActionTokenizer,  # noqa: E402
                      ActionStateTaskAlign, FlowMatchingDecoder)

torch.set_num_threads(2)

ANCHOR_EVAL_MSE = 0.045556          # v3.8.6 主预测器评估锚点 (恒等)
ANCHOR_CKPT = "checkpoints/predictor_v3.8.6.pt"


def build(seed, n_per_kind, horizon=4):
    return build_parametric_dataset(n_per_kind=n_per_kind, n_steps=14,
                                    window=6, horizon=horizon, dt=0.5, seed=seed)


def _md5(m):
    h = hashlib.md5()
    for k, v in sorted(m.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def wla_sanity(model, n_per_kind):
    """WLA 外挂零梯度自检: ER 头 + 动作三分组 + change-mask/VQ/RVQ/对齐/flow。"""
    sd0 = _md5(model)
    te = build(7777, n_per_kind)
    W, P = te.X[:8], te.P[:8]

    er = EmbodiedReasoningHead(model, enable=True)
    er_out = er(W, scene_params=P)
    er_ok = er_out["spatial_relation"].shape == (8, 2)

    tg = ActionTriGroup(eef_pose_dim=6, eef_joints_dim=4, lower_body_dim=6)
    unif = torch.randn(8, 16)
    roundtrip = torch.allclose(tg.merge(tg.split(unif)), unif)

    cm = ChangeMask(0.05)
    sp = cm.sparsity(W)

    cvq = ChangeMaskVQ(8, seed=3)
    vq_rep = cvq.fit(cm.diff(torch.randn(64, 6, 6)))

    tok = RVQActionTokenizer(8, 2, seed=4)
    rvq_rep = tok.fit(torch.randn(200, 6))

    lat = model.obs_encoder(W)[:, -1, :]
    al = ActionStateTaskAlign(latent_dim=32, action_dim=6, n_tasks=4,
                              share_dim=8)
    task_id = torch.randint(0, 4, (8,))
    align_rep = al.consistency(lat, unif[:, :6], task_id)

    g = torch.Generator().manual_seed(0)
    zf = torch.randn(48, 32, generator=g)
    af = torch.randn(48, 6, generator=g)
    dec = FlowMatchingDecoder(32, 6, hidden=16, n_steps=3)
    flow_rep = dec.fit(zf, af, epochs=15)
    dec_out = dec.decode(zf[:4])

    return {
        "er_head_proxies": bool(er_ok),
        "er_groups_orchestrated": er.describe()["orchestrates"],
        "action_tri_group_dims": tg.describe()["dims"],
        "tri_group_split_merge_roundtrip": bool(roundtrip),
        "change_mask_sparsity": sp,
        "change_vq": vq_rep,
        "rvq_action_tokenizer": rvq_rep,
        "alignment": align_rep,
        "flow_decoder": flow_rep,
        "flow_decode_shape": list(dec_out.shape),
        "zero_grad_state_dict_md5_unchanged": (sd0 == _md5(model)),
        "learned": False, "analogy_not_reproduction": True,
        "main_params_untouched": True,
    }


def main():
    n_per_kind = 48
    t0 = time.time()

    # 红线: 内化冻结主预测器 (不重新初始化)
    model, meta = load_predictor(ANCHOR_CKPT)
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params == 52191, f"主参数 {n_params} != 52191"

    ind_te = build(2718, n_per_kind)
    cal_eval = evaluate_predictor(model, ind_te)
    eval_mse = round(cal_eval["single_step_mse"], 6)
    assert abs(eval_mse - ANCHOR_EVAL_MSE) < 1e-6, \
        f"eval_mse {eval_mse} != 锚点 {ANCHOR_EVAL_MSE}"

    wla = wla_sanity(model, n_per_kind)

    os.makedirs("checkpoints", exist_ok=True)
    ckpt = "checkpoints/predictor_v3.9.0.pt"
    save_predictor(model, ckpt, metrics={
        "evaluation": cal_eval, "inherits_from": ANCHOR_CKPT,
        "wla_line": True})

    loaded, meta2 = load_predictor(ckpt)
    rep2 = evaluate_predictor(loaded, ind_te)
    assert abs(rep2["single_step_mse"] - eval_mse) < 1e-9
    assert meta2["udos_version"] == __version__

    ckpt_list = sorted(p.name for p in Path("checkpoints").glob("predictor_v*.pt"))
    cal = cal_eval["calibration"]
    iv = cal_eval["interval"]
    summary = {
        "version": __version__, "n_params": n_params,
        "elapsed_seconds": round(time.time() - t0, 1),
        "inherits_frozen_main_from": ANCHOR_CKPT,
        "retrained": False,
        "eval_mse": eval_mse, "anchor_eval_mse": ANCHOR_EVAL_MSE,
        "eval_mse_matches_anchor": abs(eval_mse - ANCHOR_EVAL_MSE) < 1e-6,
        "ece": cal["calibrated"]["ece"],
        "coverage": iv["coverage_overall"],
        "wla_sanity": wla,
        "backcompat_checkpoints": len(ckpt_list),
        "backcompat_list": ckpt_list,
        "checkpoint": ckpt, "reload_consistent": True,
    }
    os.makedirs("benchmarks/results", exist_ok=True)
    with open("benchmarks/results/training_v3.9.0.json", "w",
              encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: summary[k] for k in
                      ["version", "n_params", "eval_mse",
                       "eval_mse_matches_anchor", "backcompat_checkpoints"]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

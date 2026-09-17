"""
v2.7.0 MPC 式候选动作优选 (udos.policy.MPCActionSelector) 单元测试
=====================================================================
锚点纪律:
    * 空候选集 => no_valid_action=True, best_action=None;
    * 已知最优动作 (自定义 objective_reward) 被正确选为 best;
    * 风险惩罚单调性: OOD 扰动动作 risk 更高 => score 更低;
    * 安全边界过滤: 越下界动作 safe=False 且被 safety_penalty 扣分;
    * save/load 兼容: 纯前向不改 predictor, select 前后 predict_next 逐位一致,
      checkpoint 重新加载后 selector 仍可用。
"""
import pytest
import torch

from udos import __version__
from udos.persistence import load_predictor, save_predictor
from udos.policy import MPCActionSelector
from udos.dynamics import RAW_DIM, build_parametric_dataset

CKPT = "checkpoints/predictor_v2.7.0.pt"


@pytest.fixture(scope="module")
def predictor():
    model, _ = load_predictor(CKPT)
    return model


@pytest.fixture(scope="module")
def window_batch():
    ds = build_parametric_dataset(n_per_kind=8, n_steps=14, window=6,
                                  horizon=4, dt=0.5, seed=401)
    return ds.X[:1], ds.P[:1]


def test_version():
    assert __version__ == "5.5.5"


def test_empty_candidates_guard(predictor, window_batch):
    wb, pb = window_batch
    sel = MPCActionSelector(predictor, horizon=2)
    out = sel.select(wb, scene_params=pb, candidate_actions=[])
    assert out["no_valid_action"] is True
    assert out["best_action"] is None
    assert out["best_score"] is None
    assert out["ranked_actions"] == []
    # None 也视为空
    out2 = sel.select(wb, scene_params=pb, candidate_actions=None)
    assert out2["no_valid_action"] is True


def test_known_best_action_selected(predictor, window_batch):
    wb, pb = window_batch
    # 自定义奖励: 直接读动作自带的 _reward, 屏蔽 rollout/风险差异
    def reward_fn(pred, action):
        return float(action["_reward"])
    actions = [
        {"_reward": 0.1},
        {"_reward": 0.9},     # 已知最优
        {"_reward": 0.5},
    ]
    sel = MPCActionSelector(predictor, horizon=2, lambda_risk=1.0,
                            safety_penalty=10.0, objective_reward=reward_fn)
    out = sel.select(wb, scene_params=pb, candidate_actions=actions)
    assert out["no_valid_action"] is False
    assert out["best_index"] == 1
    assert out["best_action"]["_reward"] == 0.9
    # 全排序按 score 降序
    scores = [r["score"] for r in out["ranked_actions"]]
    assert scores == sorted(scores, reverse=True)
    assert out["ranked_actions"][0]["action"]["_reward"] == 0.9


def test_risk_penalty_monotonicity(predictor, window_batch):
    wb, pb = window_batch
    # 奖励恒 0 => score = -lambda*risk - safety_penalty*(!safe)
    zero_reward = lambda pred, action: 0.0
    # 动作 A: 无扰动 (分布内, risk 低); 动作 B: 末帧放大 5x (OOD, risk 高)
    actions = [
        {},
        {"state_perturbation": torch.zeros(RAW_DIM).index_fill_(
            0, torch.tensor(0), 100.0)},   # 沿 x 方向巨大扰动 -> OOD
    ]
    sel = MPCActionSelector(predictor, horizon=2, lambda_risk=5.0,
                            safety_penalty=10.0, objective_reward=zero_reward)
    out = sel.select(wb, scene_params=pb, candidate_actions=actions)
    ranked = {r["index"]: r for r in out["ranked_actions"]}
    # B (index 1) 风险显著高于 A (index 0)
    assert ranked[1]["risk"] > ranked[0]["risk"]
    # 风险惩罚单调: A 分数高于 B
    assert ranked[0]["score"] > ranked[1]["score"]


def test_safety_boundary_penalizes(predictor, window_batch):
    wb, pb = window_batch
    saved = predictor.residual_quantiles
    # 位置维半宽极大 => 下界 = median - 10; 越下界提议目标态 safe=False
    predictor.residual_quantiles = [torch.full((RAW_DIM,), 10.0)]
    try:
        zero_reward = lambda pred, action: 0.0
        median = predictor.predict_next(wb, scene_params=pb)[0]
        # A: 提议目标态 = median (位置维距下界 ~10 > 0 => safe)
        cand_ok = median.clone()
        # B: 提议目标态位置维压到 median - 50 => 距下界 -40 < 0 => safe=False
        cand_bad = median.clone()
        cand_bad[:3] = median[:3] - 50.0
        actions = [{"candidate_state": cand_ok},
                   {"candidate_state": cand_bad}]
        sel = MPCActionSelector(predictor, horizon=1, lambda_risk=0.0,
                                safety_penalty=100.0,
                                objective_reward=zero_reward)
        out = sel.select(wb, scene_params=pb, candidate_actions=actions)
        ranked = {r["index"]: r for r in out["ranked_actions"]}
        assert ranked[0]["safe"] is True
        assert ranked[1]["safe"] is False
        # 越界动作被扣 100
        assert ranked[0]["score"] == pytest.approx(
            ranked[0]["reward"] - 0.0, abs=1e-4)
        assert ranked[1]["score"] == pytest.approx(
            ranked[1]["reward"] - 100.0, abs=1e-4)
        assert ranked[0]["score"] > ranked[1]["score"]
    finally:
        predictor.residual_quantiles = saved


def test_deterministic(predictor, window_batch):
    wb, pb = window_batch
    actions = [{}, {"scene_param": pb.reshape(-1) * 1.0},
               {"state_perturbation": torch.zeros(RAW_DIM)}]
    sel = MPCActionSelector(predictor, horizon=2, lambda_risk=1.0)
    a = sel.select(wb, scene_params=pb, candidate_actions=actions)
    b = sel.select(wb, scene_params=pb, candidate_actions=actions)
    assert a["best_index"] == b["best_index"]
    assert a["ranked_actions"] == b["ranked_actions"]


def test_selector_does_not_modify_predictor(predictor, window_batch, tmp_path):
    """select 前后 predict_next 逐位一致, 且 save/load 后仍可用。"""
    wb, pb = window_batch
    before = predictor.predict_next(wb, scene_params=pb).clone()
    actions = [{}, {"state_perturbation": torch.zeros(RAW_DIM)}]
    sel = MPCActionSelector(predictor, horizon=2, lambda_risk=1.0)
    sel.select(wb, scene_params=pb, candidate_actions=actions)
    after = predictor.predict_next(wb, scene_params=pb)
    assert torch.allclose(before, after, atol=1e-6)

    # save/load 兼容: 重新落盘再加载, selector 对新 predictor 仍工作
    ck = tmp_path / "mpc_compat.pt"
    save_predictor(predictor, str(ck))
    loaded, meta = load_predictor(str(ck))
    sel2 = MPCActionSelector(loaded, horizon=2, lambda_risk=1.0)
    out = sel2.select(wb, scene_params=pb, candidate_actions=actions)
    assert out["no_valid_action"] is False
    assert out["best_action"] is not None
    assert meta["udos_version"] == __version__

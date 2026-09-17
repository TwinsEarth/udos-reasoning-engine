"""
v5.4.6 主预测链接线 (GPM 场景参数 -> 训练过场景门的主预测员 CTM)
================================================================
v5.4.5 建好了场景参数通道, 但 reason() 仍只把位置/速度窗口喂给训练好的
PhysicsPredictor.rollout(), 场景参数被丢弃。本版本把 5.4.5 的 SceneParams
传进主预测员, 使"GPM 记录场景 -> CTM 据场景推算未来"名副其实。

契约:
  1. reason 的 future_states 与直接 predictor.rollout(raw,H,scene_params=P)
     数值一致 (主链确实消费了场景参数);
  2. 同一 spring 窗口, 带正确角频率比场景盲的多步位置误差显著更低
     (发布 checkpoint 自带消融 condition_gain≈15x, 独立集复现);
  3. 场景无任何场景参数时, 主链逐位等价旧版 rollout(raw,H), predictor 未被
     场景调制 (向后兼容);
  4. 结果透出 predictor_conditioned / scene_params / scene_params_source;
  5. attributes 通道与 metadata 通道等价到达主预测员。
"""

from pathlib import Path

import pytest
import torch

from udos.ctm_engine import CTMConfig
from udos.dynamics import traj_spring
from udos.gpm_engine import GPMConfig, TinyBaseModel
from udos.pce_format import PhysicalToken, PhysicsScene
from udos.persistence import load_predictor
from udos.reasoning import UDOSReasoningEngine

CKPT = Path(__file__).resolve().parents[1] / "checkpoints" / "predictor_v4.3.9.pt"
W, H, DT = 6, 4, 0.5
OMEGA, AMP = 1.2, 1.5


@pytest.fixture(scope="module")
def predictor():
    p, _ = load_predictor(CKPT)
    p.eval()
    return p


def _engine(predictor):
    return UDOSReasoningEngine(
        CTMConfig(iterations=2, d_model=16, d_input=16, heads=2,
                  n_synch_out=8, n_synch_action=8, memory_length=4,
                  nlm_hidden=8, out_dims=16, certainty_threshold=0.0,
                  n_random_pairing_self=2),
        GPMConfig(feature_dim=16, latent_size=8, n_latents=4, lora_rank=4,
                  layer_indices=(0, 1), num_pre_head_layers=1, heads=2),
        base_model=TinyBaseModel(hidden=16, n_layers=2),
        predictor=predictor)


def _spring_scene(omega, channel="none"):
    traj = traj_spring(W + H, DT, amp=AMP, omega=omega, phi=0.0)
    meta = {"scene_params": {"spring_omega": omega}} if channel == "metadata" else {}
    scene = PhysicsScene(scene_id="spring", duration=W + H, metadata=meta)
    for i, (pos, vel) in enumerate(traj[:W]):
        attrs = {"spring_omega": omega} if channel == "attributes" else {}
        scene.add(PhysicalToken(object_id="obj-a", timestamp=i,
                                position=pos, velocity=vel, attributes=attrs))
    gt_pos = [p for p, _ in traj[W:W + H]]
    return scene, gt_pos


def _pos_mse(result, gt_pos):
    pred = torch.tensor([s["position"] for s in result.future_states])
    gt = torch.tensor(gt_pos)
    return ((pred - gt) ** 2).mean().item()


def test_reason_future_matches_direct_scene_conditioned_rollout(predictor):
    # 反例: reason 若丢弃场景参数, 其 future 与"带场景直接 rollout"不一致
    eng = _engine(predictor)
    scene, _ = _spring_scene(OMEGA, channel="metadata")
    r = eng.reason(scene, horizon=H)
    raw = eng._tokens_raw(scene)
    direct = predictor.rollout(raw, H, scene_params=torch.tensor(
        [[0.0, 0.0, OMEGA, 0.0]]))[0]
    got = torch.tensor([s["position"] + s["velocity"] for s in r.future_states])
    assert torch.allclose(got, direct, atol=1e-5)


def test_scene_conditioned_more_accurate_than_blind(predictor):
    eng = _engine(predictor)
    scene_blind, gt = _spring_scene(OMEGA, channel="none")
    scene_scene, _ = _spring_scene(OMEGA, channel="metadata")
    r_blind = eng.reason(scene_blind, horizon=H)
    r_scene = eng.reason(scene_scene, horizon=H)
    mse_blind = _pos_mse(r_blind, gt)
    mse_scene = _pos_mse(r_scene, gt)
    # 带正确隐藏角频率必须显著更准 (实测约一个数量级, 留稳健余量)
    assert mse_scene < 0.5 * mse_blind, (mse_blind, mse_scene)
    assert r_scene.predictor_conditioned is True
    assert r_blind.predictor_conditioned is False


def test_blind_path_bitidentical_to_legacy(predictor):
    # 向后兼容: 无场景参数时主链与旧版 rollout(raw,H) 逐位一致
    eng = _engine(predictor)
    scene, _ = _spring_scene(OMEGA, channel="none")
    r = eng.reason(scene, horizon=H)
    raw = eng._tokens_raw(scene)
    legacy = predictor.rollout(raw, H)[0]
    got = torch.tensor([s["position"] + s["velocity"] for s in r.future_states])
    assert torch.allclose(got, legacy, atol=1e-6)
    assert r.predictor_conditioned is False
    assert r.scene_params is None
    assert r.scene_params_source is None


def test_result_exposes_scene_params_and_source(predictor):
    eng = _engine(predictor)
    scene, _ = _spring_scene(OMEGA, channel="metadata")
    r = eng.reason(scene, horizon=H)
    assert r.predictor_conditioned is True
    assert r.scene_params_source == "metadata"
    assert len(r.scene_params) == 4
    assert abs(r.scene_params[2] - OMEGA) < 1e-6
    summary = r.summary()
    assert summary["predictor_conditioned"] is True
    assert abs(summary["scene_params"][2] - OMEGA) < 1e-6
    assert summary["scene_params_source"] == "metadata"


def test_attributes_channel_reaches_predictor_same_as_metadata(predictor):
    eng = _engine(predictor)
    scene_attr, _ = _spring_scene(OMEGA, channel="attributes")
    scene_meta, _ = _spring_scene(OMEGA, channel="metadata")
    r_attr = eng.reason(scene_attr, horizon=H)
    r_meta = eng.reason(scene_meta, horizon=H)
    assert r_attr.predictor_conditioned is True
    assert r_attr.scene_params_source == "attributes"
    a = torch.tensor([s["position"] for s in r_attr.future_states])
    m = torch.tensor([s["position"] for s in r_meta.future_states])
    assert torch.allclose(a, m, atol=1e-6)

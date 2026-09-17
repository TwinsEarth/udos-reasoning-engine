"""
v3.9.0 宇树 UnifoLM-WLA 机制类比线测试
=========================================
覆盖: EmbodiedReasoningHead / ActionTriGroup / ChangeMask / ChangeMaskVQ /
      RVQActionTokenizer / ActionStateTaskAlign / FlowMatchingDecoder。
纪律: 外挂零梯度 (主 predictor md5 不变)、空/非法显式 ValueError、版本断言。
"""
import hashlib

import pytest
import torch

from udos import (
    __version__,
    CTMConfig,
    PhysicsPredictor,
    EmbodiedReasoningHead,
    ActionTriGroup,
    ChangeMask,
    ChangeMaskVQ,
    RVQActionTokenizer,
    ActionStateTaskAlign,
    FlowMatchingDecoder,
)


def _md5(m):
    h = hashlib.md5()
    for k, v in sorted(m.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def _predictor():
    cfg = CTMConfig(iterations=8, d_model=64, d_input=32, heads=4,
                    n_synch_out=16, n_synch_action=8, memory_length=8,
                    nlm_hidden=16, out_dims=32, certainty_threshold=0.0)
    return PhysicsPredictor(cfg, scene_param_dim=4)


def test_version():
    assert __version__ == "5.5.5"


# --------------------------------------------------------------------------- #
# EmbodiedReasoningHead
# --------------------------------------------------------------------------- #
def test_er_head_default_off_returns_empty():
    pred = _predictor()
    head = EmbodiedReasoningHead(pred, enable=False)
    out = head(torch.randn(2, 6, 6))
    assert out == {}


def test_er_head_enable_on_proxies_and_zerograd():
    pred = _predictor()
    head = EmbodiedReasoningHead(pred, enable=True)
    w = torch.randn(3, 6, 6)
    sd0 = _md5(pred)
    out = head(w)
    assert set(out) == {"spatial_relation", "target_point",
                        "trajectory_proxy", "latent_dim"}
    assert out["spatial_relation"].shape == (3, 2)
    assert out["target_point"].shape == (3, 2)
    assert out["trajectory_proxy"].shape == (3, 2)
    # 软压到 (-1,1)
    assert float(out["target_point"].abs().max()) <= 1.0
    # 零梯度: 主权重 md5 不变
    assert _md5(pred) == sd0
    d = head.describe()
    assert d["rebuild"] is False and d["no_new_trainable_params"] is True
    assert d["analogy_not_reproduction"] is True


def test_er_head_bad_window():
    head = EmbodiedReasoningHead(_predictor(), enable=True)
    with pytest.raises(ValueError):
        head(torch.randn(6))          # 1D
    with pytest.raises(ValueError):
        head(torch.randn(3, 6, 6, 6))  # 4D


# --------------------------------------------------------------------------- #
# ActionTriGroup
# --------------------------------------------------------------------------- #
def test_action_tri_group_split_merge_roundtrip():
    tg = ActionTriGroup(6, 4, 6)
    assert tg.total_dim == 16
    a = torch.randn(5, 16)
    parts = tg.split(a)
    assert parts["eef_pose"].shape == (5, 6)
    assert parts["eef_joints"].shape == (5, 4)
    assert parts["lower_body"].shape == (5, 6)
    assert torch.allclose(tg.merge(parts), a)


def test_action_tri_group_zero_joints_ok():
    tg = ActionTriGroup(6, 0, 4)
    a = torch.randn(3, 10)
    parts = tg.split(a)
    assert "eef_joints" not in parts
    assert parts["eef_pose"].shape == (3, 6)
    assert parts["lower_body"].shape == (3, 4)


def test_action_tri_group_guards():
    with pytest.raises(ValueError):
        ActionTriGroup(0, 4, 6)     # eef_pose 不可缺
    with pytest.raises(ValueError):
        ActionTriGroup(6, -1, 6)    # 负维
    tg = ActionTriGroup(6, 4, 6)
    with pytest.raises(ValueError):
        tg.split(torch.randn(3, 15))   # 维不符
    with pytest.raises(ValueError):
        tg.split(torch.tensor([]))     # 空
    with pytest.raises(ValueError):
        tg.split(torch.tensor([float("nan")] * 16).reshape(1, 16))


# --------------------------------------------------------------------------- #
# ChangeMask (dev1)
# --------------------------------------------------------------------------- #
def test_change_mask_diff_mask_sparsity():
    cm = ChangeMask(0.05)
    w = torch.randn(8, 6, 6)
    d = cm.diff(w)
    assert d.shape == (8, 6)
    m = cm.mask(w)
    assert m.shape == (8, 6)
    assert set(torch.unique(m).tolist()) <= {0.0, 1.0}
    s = cm.sparsity(w)
    assert 0.0 <= s["changed_ratio"] <= 1.0
    assert s["raw_dim"] == 6


def test_change_mask_sparse_next_identity_and_blend():
    cm = ChangeMask(0.5)   # 高阈值 => 几乎全不变
    w = torch.randn(4, 6, 6)
    ident = cm.sparse_next(w)
    assert torch.allclose(ident, w[:, -1, :])
    # 低阈值 => 几乎全用预测
    cm2 = ChangeMask(1e-6)
    p = torch.randn(4, 6)
    out = cm2.sparse_next(w, p)
    assert out.shape == (4, 6)


def test_change_mask_guards():
    cm = ChangeMask(0.05)
    with pytest.raises(ValueError):
        cm.diff(torch.randn(6))        # 1D -> 仍 1D, 非法
    with pytest.raises(ValueError):
        cm.diff(torch.randn(2, 6, 6, 6))  # 4D
    with pytest.raises(ValueError):
        cm.diff(torch.randn(3, 1, 6))   # W=1, 不足两帧
    with pytest.raises(ValueError):
        cm.sparse_next(torch.randn(4, 6, 6), torch.randn(4, 5))  # 形状不符


# --------------------------------------------------------------------------- #
# ChangeMaskVQ (dev2)
# --------------------------------------------------------------------------- #
def test_change_mask_vq_fit_report_roundtrip():
    cm = ChangeMask(0.05)
    x = cm.diff(torch.randn(128, 6, 6))
    vq = ChangeMaskVQ(8, seed=1)
    rep = vq.fit(x, steps=20)
    assert rep["codebook_size"] == 8
    assert 0.0 <= rep["utilization"] <= 1.0
    assert rep["recon_mse"] is not None and rep["recon_mse"] >= 0
    assert isinstance(rep["collapsed"], bool)
    ids = vq.encode(x[:5])
    assert ids.shape == (5,)
    rec = vq.decode(ids)
    assert rec.shape == (5, 6)


def test_change_mask_vq_guards():
    vq = ChangeMaskVQ(8)
    with pytest.raises(ValueError):
        vq.encode(torch.randn(3, 6))     # 未 fit
    with pytest.raises(ValueError):
        vq.fit(torch.randn(2, 6))        # N < k


# --------------------------------------------------------------------------- #
# RVQActionTokenizer (dev4)
# --------------------------------------------------------------------------- #
def test_rvq_fit_roundtrip_and_report():
    tok = RVQActionTokenizer(codebook_size=8, n_levels=2, seed=2)
    x = torch.randn(200, 4)
    rep = tok.fit(x, steps=20)
    assert rep["n_levels"] == 2
    assert rep["recon_mse"] >= 0
    assert len(rep["levels"]) == 2
    assert isinstance(rep["collapsed"], bool)
    ids = tok.encode(x[:10])
    assert ids.shape == (10, 2)
    rec = tok.decode(ids)
    assert rec.shape == (10, 4)


def test_rvq_guards():
    tok = RVQActionTokenizer(8, 2)
    with pytest.raises(ValueError):
        tok.encode(torch.randn(3, 4))    # 未 fit
    with pytest.raises(ValueError):
        tok.fit(torch.randn(3, 4))       # N < k


# --------------------------------------------------------------------------- #
# ActionStateTaskAlign (dev5)
# --------------------------------------------------------------------------- #
def test_align_loss_and_consistency():
    al = ActionStateTaskAlign(8, 4, n_tasks=3, share_dim=8)
    g = torch.Generator().manual_seed(0)
    z = torch.randn(30, 8, generator=g)
    a = torch.randn(30, 4, generator=g)
    t = torch.randint(0, 3, (30,), generator=g)
    loss = al.alignment_loss(z, a)
    assert -1.0 <= loss <= 2.0
    c = al.consistency(z, a, t)
    assert set(c) >= {"same_task_alignment", "cross_task_alignment", "consistent"}
    assert isinstance(c["consistent"], bool)


def test_align_guards():
    al = ActionStateTaskAlign(8, 4, 3)
    with pytest.raises(ValueError):
        al.embed_state(torch.randn(5, 7))      # 维不符
    with pytest.raises(ValueError):
        al.embed_task(torch.tensor([5]))      # task_id 越界


# --------------------------------------------------------------------------- #
# FlowMatchingDecoder (dev6)
# --------------------------------------------------------------------------- #
def test_flow_decoder_fit_decode_and_regress():
    g = torch.Generator().manual_seed(0)
    z = torch.randn(64, 8, generator=g)
    a = torch.randn(64, 4, generator=g)
    dec = FlowMatchingDecoder(8, 4, hidden=16, n_steps=3)
    rep = dec.fit(z, a, epochs=10)
    assert rep["frozen_backbone_zero_grad"] is True
    assert dec.decode(z[:5]).shape == (5, 4)
    assert dec.decode_regress(z[:5]).shape == (5, 4)
    # 确定性: 同 seed decode 逐位一致
    assert torch.equal(dec.decode(z[:3], seed=1), dec.decode(z[:3], seed=1))


def test_flow_decoder_is_external_zero_grad_to_predictor():
    pred = _predictor()
    g = torch.Generator().manual_seed(0)
    z = torch.randn(32, pred.ctm.cfg.d_input, generator=g)
    a = torch.randn(32, 4, generator=g)
    sd0 = _md5(pred)
    dec = FlowMatchingDecoder(pred.ctm.cfg.d_input, 4, 16, 3)
    dec.fit(z, a, epochs=3)
    dec.decode(z[:4])
    # 主 predictor 权重 md5 不变 (外挂, 不进主 state_dict)
    assert _md5(pred) == sd0

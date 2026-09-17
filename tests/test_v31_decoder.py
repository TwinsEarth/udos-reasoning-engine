"""
v3.1.0.dev3 节点34: TokenActionDecoder 硬/平滑解码 + 关节限位 + retargeting 集成
================================================================
锚点纪律:
    * 硬解码形状 [N, D]、有限; 平滑解码步间跳变 < 硬解码 (更连续);
    * 关节限位后处理后严格落在限位内;
    * 解码动作可直接喂给 ActionRetargeter.retarget (接口一致);
    * 未拟合 tokenizer / 非法 joint_limits / interp_steps<1 守卫。
analogy, not reproduction —— 合成动作, 非真机控制。
"""
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from udos.action_piece import ActionPieceTokenizer, TokenActionDecoder  # noqa: E402
from udos.retargeting import MorphologyConfig, ActionRetargeter  # noqa: E402


def make_tokenizer():
    g = torch.Generator().manual_seed(5)
    actions = torch.randn(300, 6, generator=g)
    return ActionPieceTokenizer(action_dim=6, codebook_size=8, seed=42).fit(actions)


def test_hard_decode_shape_finite():
    tok = make_tokenizer()
    dec = TokenActionDecoder(tok)
    toks = torch.tensor([0, 3, 5, 2, 7, 1])
    a = dec.hard_decode(toks)
    assert a.shape == (6, 6)
    assert torch.isfinite(a).all()


def test_smoothing_reduces_jitter():
    tok = make_tokenizer()
    dec = TokenActionDecoder(tok)
    toks = torch.tensor([0, 4, 1, 6, 2, 5, 3, 7])
    hard = dec.hard_decode(toks)
    smooth = dec.smooth_decode(toks, interp_steps=4)
    # 平滑轨迹更密
    assert smooth.size(0) > hard.size(0)
    # 平滑后相邻步跳变更小
    assert TokenActionDecoder.continuity(smooth) < \
        TokenActionDecoder.continuity(hard)


def test_interp_steps_one_equals_hard():
    tok = make_tokenizer()
    dec = TokenActionDecoder(tok)
    toks = torch.tensor([0, 3, 5])
    hard = dec.hard_decode(toks)
    s1 = dec.smooth_decode(toks, interp_steps=1)
    assert torch.allclose(hard, s1, atol=1e-6)


def test_joint_limits_clamp():
    tok = make_tokenizer()
    # 紧限位, 强制裁剪
    limits = torch.tensor([[-0.5, 0.5]] * 6, dtype=torch.float32)
    dec = TokenActionDecoder(tok, joint_limits=limits)
    toks = torch.tensor([0, 4, 2, 6, 1])
    a = dec.hard_decode(toks)
    assert dec.within_limits(a)
    assert bool((a >= -0.5 - 1e-6).all() and (a <= 0.5 + 1e-6).all())
    sa = dec.smooth_decode(toks, interp_steps=3)
    assert dec.within_limits(sa)


def test_retargeting_integration():
    tok = make_tokenizer()
    dec = TokenActionDecoder(tok)
    toks = torch.tensor([0, 3, 5, 2, 7])
    actions = dec.hard_decode(toks)          # [5, 6] 源形态动作
    src = MorphologyConfig(dof=6, control_freq=60.0,
                           joint_limits=[[-2.0, 2.0]] * 6, name="p")
    tgt = MorphologyConfig(dof=4, control_freq=120.0,
                           joint_limits=[[-0.5, 0.5]] * 4, name="g")
    rt = ActionRetargeter(src, tgt)
    out = rt.retarget(actions)
    assert out.shape == (5, 4)
    assert bool((out >= -0.5 - 1e-6).all() and (out <= 0.5 + 1e-6).all())


def test_unfitted_tokenizer_guard():
    raw = ActionPieceTokenizer(action_dim=6, codebook_size=4)
    with pytest.raises(ValueError):
        TokenActionDecoder(raw)


def test_bad_limits_and_interp_guard():
    tok = make_tokenizer()
    with pytest.raises(ValueError):
        TokenActionDecoder(tok, joint_limits=torch.tensor([[0.0]]))  # 形状错
    with pytest.raises(ValueError):
        TokenActionDecoder(tok, joint_limits=torch.tensor(
            [[1.0, 0.0]] * 6))                                        # lo>hi
    dec = TokenActionDecoder(tok)
    with pytest.raises(ValueError):
        dec.smooth_decode(torch.tensor([0, 1]), interp_steps=0)


def test_single_token_continuity_zero():
    tok = make_tokenizer()
    dec = TokenActionDecoder(tok)
    a = dec.hard_decode(torch.tensor([3]))
    assert a.shape == (1, 6)
    assert TokenActionDecoder.continuity(a) == 0.0

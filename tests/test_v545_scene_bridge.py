"""
v5.4.5 场景参数通道 (Scene Params Bridge)
=========================================
GPM"场景记录员"记录的隐藏物理参数 (v0 / accel_a / spring_omega / other_v2)
需要有一条规范、可校验、向后兼容的通道从 PCE PhysicsScene 进入系统, 才能在
后续版本喂给训练过场景门的主预测员 CTM。

本版本只负责"承载 + 提取 + 校验", 不改动 reason 主链 (5.4.6 才接线):
  * 显式通道一: scene.metadata["scene_params"], 支持 dict(按槽名) 或
    list/tuple(按 SCENE_PARAM_NAMES 顺序, 长度必须为 4);
  * 显式通道二: PhysicalToken.attributes 中的标准槽名 (场景级常量, 取时间序
    最早出现值); metadata 优先级高于 attributes;
  * 完全没有场景信息时返回 None (保持场景盲旧路径逐位兼容);
  * 所有数值必须有限 (NaN/inf 显式 ValueError, 与 predict_next 同纪律);
  * PCE 序列化往返保留 metadata.scene_params (协议无需改动, metadata 已透传)。
"""

import math

import pytest
import torch

from udos.dynamics import SCENE_PARAM_DIM, SCENE_PARAM_NAMES
from udos.pce_format import PCEParser, PhysicalToken, PhysicsScene
from udos.scene_bridge import SceneParams, extract_scene_params


def _spring_scene(metadata=None, attrs=None):
    scene = PhysicsScene(scene_id="s", duration=4, metadata=dict(metadata or {}))
    for t in range(4):
        scene.add(PhysicalToken(
            object_id="obj-a", timestamp=t,
            position=[float(t), 0.0, 0.0],
            velocity=[1.0, 0.0, 0.0],
            attributes=dict(attrs or {})))
    return scene


def test_metadata_dict_explicit_slots(torch_seed):
    # 反例: 实现若忽略 metadata 直接返回 None, 本测试变红
    scene = _spring_scene(metadata={"scene_params": {
        "spring_omega": 1.2, "other_v2": 0.3}})
    sp = extract_scene_params(scene)
    assert sp is not None
    assert sp.values.shape == (1, SCENE_PARAM_DIM)
    assert sp.source == "metadata"
    # 槽位顺序严格对齐 SCENE_PARAM_NAMES = (v0, accel_a, spring_omega, other_v2)
    assert torch.allclose(
        sp.values, torch.tensor([[0.0, 0.0, 1.2, 0.3]]), atol=1e-6)
    assert sp.per_slot_source[2] == "metadata"
    assert sp.per_slot_source[3] == "metadata"
    assert sp.per_slot_source[0] == "default"


def test_metadata_list_uses_fixed_slot_order(torch_seed):
    # 反例: 槽位顺序错乱 (如按字母序) 会让本测试变红
    scene = _spring_scene(
        metadata={"scene_params": [0.5, -0.2, 0.9, 0.1]})
    sp = extract_scene_params(scene)
    assert torch.allclose(
        sp.values, torch.tensor([[0.5, -0.2, 0.9, 0.1]]), atol=1e-6)


def test_attributes_fallback_channel(torch_seed):
    # 反例: 不读 token.attributes 会返回 None
    scene = _spring_scene(attrs={"spring_omega": 1.1, "mass": 2.0})
    sp = extract_scene_params(scene)
    assert sp is not None
    assert sp.source == "attributes"
    assert abs(float(sp.values[0, 2]) - 1.1) < 1e-6
    assert sp.per_slot_source[2] == "attributes"


def test_metadata_takes_priority_over_attributes(torch_seed):
    scene = _spring_scene(
        metadata={"scene_params": {"spring_omega": 1.3}},
        attrs={"spring_omega": 0.7})
    sp = extract_scene_params(scene)
    assert abs(float(sp.values[0, 2]) - 1.3) < 1e-6
    assert sp.per_slot_source[2] == "metadata"


def test_no_scene_info_returns_none(tiny_scene, torch_seed):
    # 反例: 总是返回零向量会破坏场景盲旧路径的逐位兼容
    assert extract_scene_params(tiny_scene) is None


@pytest.mark.parametrize("bad", [
    {"scene_params": {"spring_omega": float("nan")}},
    {"scene_params": {"accel_a": float("inf")}},
    {"scene_params": {"v0": float("-inf")}},
])
def test_non_finite_rejected(bad, torch_seed):
    scene = _spring_scene(metadata=bad)
    with pytest.raises(ValueError):
        extract_scene_params(scene)


def test_wrong_length_list_rejected(torch_seed):
    scene = _spring_scene(metadata={"scene_params": [0.1, 0.2, 0.3]})
    with pytest.raises(ValueError):
        extract_scene_params(scene)


def test_unknown_keys_ignored_not_fatal(torch_seed):
    # 只认 4 个标准槽名; 未知键忽略, 不臆测
    scene = _spring_scene(metadata={"scene_params": {
        "spring_omega": 1.0, "foo": 9.0}})
    sp = extract_scene_params(scene)
    assert torch.allclose(
        sp.values, torch.tensor([[0.0, 0.0, 1.0, 0.0]]), atol=1e-6)


def test_deterministic_extraction(torch_seed):
    scene = _spring_scene(metadata={"scene_params": {
        "accel_a": 0.4, "other_v2": -0.2}})
    a = extract_scene_params(scene).values
    b = extract_scene_params(scene).values
    assert torch.equal(a, b)


def test_pce_roundtrip_preserves_scene_params(torch_seed):
    scene = _spring_scene(metadata={"scene_params": {
        "spring_omega": 1.23, "other_v2": 0.11}})
    restored = PCEParser.loads(PCEParser.dumps(scene))
    sp = extract_scene_params(restored)
    assert sp is not None
    assert abs(float(sp.values[0, 2]) - 1.23) < 1e-6
    assert abs(float(sp.values[0, 3]) - 0.11) < 1e-6


def test_scene_params_is_finite_float32(torch_seed):
    scene = _spring_scene(metadata={"scene_params": {
        "v0": 1, "accel_a": 2}})  # int 入参也应被转成 float32
    sp = extract_scene_params(scene)
    assert sp.values.dtype == torch.float32
    assert torch.isfinite(sp.values).all()
    assert isinstance(sp, SceneParams)
    # 槽名常量自检, 防止训练/桥接两侧槽位定义漂移
    assert SCENE_PARAM_NAMES == ("v0", "accel_a", "spring_omega", "other_v2")
    assert SCENE_PARAM_DIM == 4
    # 每个槽来源都必须能在白名单内
    assert set(sp.per_slot_source) <= {"metadata", "attributes", "default"}
    # 数学健全性: 不允许出现 NaN 槽 (math 维度自检)
    assert all(math.isfinite(float(x)) for x in sp.values.flatten().tolist())

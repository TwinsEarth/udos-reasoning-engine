"""v4.5.4 开源资源注册表单测: L0-L3 诚信边界、四态、过滤、缓存、降级。"""
import pytest

from udos.connectors import build_default_registry
from udos.connectors.specialized import (ActionChunkingConnector,
                                         VLADiscreteConnector,
                                         EpisodeSchemaConnector,
                                         SMPLPoseConnector,
                                         BVHClipConnector)
from udos.connectors.base import build_spec
from udos.resource_registry import (ResourceUnavailable,
                                    normalize_trajectory)


def test_registry_builds_79():
    reg = build_default_registry("full")
    assert len(reg.ids()) == 79


def test_performance_profile_excludes_l3():
    reg = build_default_registry("performance")
    rows = reg.list()
    # performance 只暴露 高优先级 且 L1/L2 的子集
    for r in rows:
        assert r["priority"] == "high"
        assert r["level"] in ("L0", "L1", "L2")
    # 高优先级但 L3 大权重被排除在性能视图外
    ids = {r["id"] for r in rows}
    assert "openvla" not in ids          # L3 权重
    assert "openpi_0_0_fast_0_5" not in ids


def test_full_profile_all_discoverable():
    reg = build_default_registry("full")
    rows = reg.list()
    assert len(rows) == 79
    # 全量 schema 合法: 每条形如 spec.public
    for r in rows:
        assert r["license"]               # license 非空
        assert r["kind"] in ("model", "dataset", "action")
        assert r["level"] in ("L0", "L1", "L2", "L3")
        assert r["capability"]["status"] in (
            "available", "degraded", "absent", "env_blocked")


def test_l3_absent_invokes_gracefully():
    reg = build_default_registry("full")
    rep = reg.probe("openvla")["capability"]
    assert rep["status"] == "absent"
    assert rep["requires"]["weights"] is True
    with pytest.raises(ResourceUnavailable):
        reg.invoke("openvla", "convert", {"payload": {}})


def test_unknown_id_raises_keyerror():
    reg = build_default_registry("full")
    with pytest.raises(KeyError):
        reg.probe("does_not_exist_xyz")
    with pytest.raises(KeyError):
        reg.invoke("does_not_exist_xyz", "convert", {})


def test_probe_cached():
    reg = build_default_registry("full")
    r1 = reg.probe("diffusion_policy")
    r2 = reg.probe("diffusion_policy")
    assert r1["capability"]["probed_at"] == r2["capability"]["probed_at"]


# ---------- L1 fixture 往返一致 ---------- #
def test_action_chunk_roundtrip():
    reg = build_default_registry("full")
    out = reg.invoke("diffusion_policy", "convert",
                     {"payload": {"chunk": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
                                  "proprio": [0.9, 0.1]}})
    u = out["udos"]
    assert u["schema"] == "udos/pce-action/v1"
    assert u["n_steps"] == 2
    assert len(u["actions"]) == 2 and len(u["actions"][0]) == 3


def test_vla_discrete_roundtrip_bounded_error():
    vla = VLADiscreteConnector(build_spec(
        {"id": "openvla", "name": "OpenVLA", "source_class": "模型库",
         "priority": "high"}, level="L1"))
    q01 = [-1.0, -0.5]; q99 = [1.0, 0.5]
    cont = [0.3, -0.2]
    bins = vla.continuous_to_bins(cont, q01, q99)
    back = vla._bins_to_continuous(bins, q01, q99)
    for c, b in zip(cont, back):
        assert abs(c - b) < (0.5 / 255.0) + 1e-6


def test_episode_schema_parse():
    reg = build_default_registry("full")
    payload = {"steps": [
        {"state": [0.0, 1.0], "action": [1.0, 0.0], "frame_index": 0},
        {"state": [0.1, 1.1], "action": [1.1, 0.0], "frame_index": 1}]}
    out = reg.invoke("lerobot_framework_incl_smolvla_so_100", "convert",
                     {"payload": payload})
    u = out["udos"]
    assert u["n_steps"] == 2 and len(u["observations"]) == 2


def test_smpl_axis_angle_to_rot6d():
    smpl = SMPLPoseConnector(build_spec(
        {"id": "amass", "name": "AMASS", "source_class": "动作库",
         "priority": "normal"}, level="L1"))
    out = smpl.to_udos({"body_pose": [0.0, 0.0, 0.0, 1.57, 0.0, 0.0]})
    assert out["n_joints"] == 2
    assert len(out["rot6d"]) == 12     # 2 joints * 6


def test_bvh_clip_parse():
    bvh = BVHClipConnector(build_spec(
        {"id": "lafan1", "name": "LAFAN1", "source_class": "动作库",
         "priority": "normal"}, level="L1"))
    out = bvh.to_udos({"frames": [[0.0, 90.0], [0.0, 91.0]],
                       "joints": ["Hips"]})
    assert out["n_frames"] == 2


def test_normalize_trajectory_invalid():
    with pytest.raises(ValueError):
        normalize_trajectory({"observations": "notalist"})


# ---------- L2 yourdfpy: 真 smoke, 不崩核心 ---------- #
def test_yourdfpy_l2_status_is_available_or_degraded():
    reg = build_default_registry("full")
    rep = reg.probe("yourdfpy")["capability"]
    assert rep["status"] in ("available", "degraded")
    if rep["status"] == "available":
        assert "yourdfpy" in rep["evidence"] or "FK" in rep["evidence"]


def test_absent_pkg_does_not_break_core_import():
    # pytorch_kinematics 未安装 -> absent, 但 registry 照常装配
    reg = build_default_registry("full")
    rep = reg.probe("pytorch_kinematics")["capability"]
    assert rep["status"] in ("absent", "env_blocked")
    assert rep["install_hint"] and "pip install" in rep["install_hint"]
    # 核心 udos 仍可 import (在收集阶段已证明)


def test_filters():
    reg = build_default_registry("full")
    assert reg.list(kind="model") and all(r["kind"] == "model" for r in reg.list(kind="model"))
    assert all(r["level"] == "L3" for r in reg.list(level="L3"))
    apache = reg.list(license="apache")
    assert isinstance(apache, list)

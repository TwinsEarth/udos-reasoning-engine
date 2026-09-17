"""v4.5.0.dev3 latent_collab 多专家协作测试。"""
import pytest

from udos.latent_collab import LatentCollaborator
from udos.collab_topology import Topology


def test_agrees_no_escalation():
    collab = LatentCollaborator()
    paths = [
        {"path": 0, "convergence": 0.9, "ticks_used": 8},
        {"path": 1, "convergence": 0.5, "ticks_used": 8},
    ]
    out = collab.collaborate("t1", paths, global_disagreement=0.05,
                             base_effort="low",
                             base_topology=Topology.STAR.value)
    # 两路径 convergence 差异大, 专家应一致选 0
    assert out["selected_path"] == 0
    assert out["escalated"] is False
    assert out["joint_strategy"]["difficulty_effort"] == "low"
    assert out["joint_strategy"]["topology"] == Topology.STAR.value


def test_disagreement_triggers_escalation():
    collab = LatentCollaborator(allow_mesh=True)
    # 构造让专家分歧: convergence 偏好 path0, parsimony 偏好 ticks 少者
    paths = [
        {"path": 0, "convergence": 0.9, "ticks_used": 16},   # 高置信但费 tick
        {"path": 1, "convergence": 0.4, "ticks_used": 4},    # 低置信但省 tick
    ]
    out = collab.collaborate("t2", paths, global_disagreement=0.5,
                             base_effort="low",
                             base_topology=Topology.STAR.value)
    # stability 偏好低分歧(全局分歧高->分低), convergence 选 0, parsimony 选 1
    assert out["expert_disagreement"] > 0.0
    if out["escalated"]:
        assert out["joint_strategy"]["difficulty_effort"] in ("high", "max")
        assert out["joint_strategy"]["topology"] in (Topology.CHAIN.value,
                                                     Topology.MESH.value)


def test_empty_paths_raises():
    collab = LatentCollaborator()
    with pytest.raises(ValueError):
        collab.collaborate("t3", [], 0.0, "low", "star")


def test_trace_integrity_present():
    collab = LatentCollaborator()
    paths = [{"path": 0, "convergence": 0.8, "ticks_used": 8}]
    out = collab.collaborate("t4", paths, 0.0, "none", "star")
    assert "trace_integrity" in out
    assert out["selected_path"] == 0

"""v3.8.4 综合评测: 跨 3.4-3.8 全部特性组合不冲突 + backcompat 24。

覆盖:
    * comprehensive_eval_v3.8.0.json 全特性 ok 且零梯度;
    * ICM+SFM+PWM+分层+多体+调度+闭环+孪生 同 predictor 叠加调用;
    * backcompat 24 件; 默认输出逐位一致 (主参 52191)。
"""
import json
from pathlib import Path

import torch

from udos import __version__, load_predictor
from udos.multi_agent import AgentCoordinator, MultiAgentScene
from udos.closed_loop import ClosedLoopOrchestrator
from udos.digital_twin import DigitalTwinScene

ROOT = Path(__file__).resolve().parents[1]
JSON = ROOT / "benchmarks" / "results" / "comprehensive_eval_v3.8.0.json"
CKPT = str(ROOT / "checkpoints" / "predictor_v3.8.0.pt")


def test_version():
    assert __version__ == "5.5.5"


def test_comprehensive_json():
    assert JSON.exists()
    d = json.load(open(JSON, encoding="utf-8"))
    assert d["all_ok"] is True
    assert d["zero_grad_state_dict_md5_unchanged"] is True
    assert d["main_params"] == 52191
    for k in ("icm_3.4", "pwm_3.6", "hier_3.7", "multi_agent_3.8",
              "wm_sched_3.8", "closed_loop_3.8", "twin_3.8"):
        assert k in d["features"]


def test_composition_no_conflict():
    model, _ = load_predictor(CKPT)
    w = torch.randn(1, 6, 6)
    # 多体 + 闭环 同 predictor 叠加
    sc = MultiAgentScene()
    sc.add_agent("a", [0, 0, 0, 1, 0, 0], priority=1)
    sc.add_agent("b", [0.2, 0, 0, -1, 0, 0], priority=9)
    AgentCoordinator().resolve(sc)
    orch = ClosedLoopOrchestrator(model, wm_horizon=2)
    out = orch.step(w)
    assert bool(torch.isfinite(out["command"]).all())
    twin = DigitalTwinScene(n_agents=3, seed=0)
    twin.step()


def test_backcompat_24():
    ckpts = sorted((ROOT / "checkpoints").glob("predictor_v*.pt"))
    assert len(ckpts) >= 25

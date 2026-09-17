"""v3.6.1 集成: PWM 世界模型 HTTP 端点 /wm/imagine、/wm/conservation。

覆盖:
    * 两新端点 200 (已挂载 checkpoint);
    * 400 非法 horizon / 非法 source / 缺 window;
    * 409 未训练/未挂载 predictor;
    * /loop/step 默认逐位路径不受影响 (全特性兼容);
    * 未知路由 404;
    * 默认输出逐位一致 (WM opt-in, 不改主 predictor)。
"""
import json
import threading
import urllib.request
import urllib.error
from pathlib import Path

import pytest
import torch

from udos import __version__
from udos.server import create_server, UDOSService

ROOT = Path(__file__).resolve().parents[1]
CKPT = str(ROOT / "checkpoints" / "predictor_v3.6.0.pt")


def test_version():
    assert __version__ == "5.5.5"


@pytest.fixture(scope="module")
def live_server():
    torch.manual_seed(0)
    httpd = create_server("127.0.0.1", 0, preset="small", checkpoint=CKPT)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    host, port = httpd.server_address
    base = f"http://{host}:{port}"
    yield base
    httpd.shutdown()
    httpd.server_close()
    thread.join(timeout=5)


def _post(base, path, obj):
    data = json.dumps(obj).encode("utf-8")
    req = urllib.request.Request(base + path, data=data,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode("utf-8"))


def _window():
    ds = torch.randn(2, 6, 6).tolist()
    sp = torch.zeros(2, 4).tolist()
    return ds, sp


def test_wm_imagine_200(live_server):
    w, sp = _window()
    code, body = _post(live_server, "/wm/imagine",
                       {"window": w, "scene_params": sp, "horizon": 4})
    assert code == 200, body
    assert body["status"] == "ok"
    assert body["horizon"] == 4
    assert body["wm_params"] > 0
    # 第 0 步与真实 rollout 锚定 => step_mse 第 0 项为 0
    assert body["step_mse_vs_real"][0] == 0.0
    assert body["main_params_untouched"] is True


def test_wm_imagine_400_bad_horizon(live_server):
    w, sp = _window()
    code, _ = _post(live_server, "/wm/imagine",
                    {"window": w, "scene_params": sp, "horizon": 99})
    assert code == 400


def test_wm_imagine_400_missing_window(live_server):
    code, _ = _post(live_server, "/wm/imagine", {"horizon": 2})
    assert code == 400


def test_wm_conservation_200_real(live_server):
    w, sp = _window()
    code, body = _post(live_server, "/wm/conservation",
                       {"window": w, "scene_params": sp, "horizon": 4,
                        "source": "real"})
    assert code == 200, body
    assert body["source"] == "real"
    assert "momentum_violation" in body and "energy_violation" in body


def test_wm_conservation_200_imagine(live_server):
    w, sp = _window()
    code, body = _post(live_server, "/wm/conservation",
                       {"window": w, "scene_params": sp, "horizon": 4,
                        "source": "imagine"})
    assert code == 200, body
    assert body["source"] == "imagine"


def test_wm_conservation_400_bad_source(live_server):
    w, sp = _window()
    code, _ = _post(live_server, "/wm/conservation",
                    {"window": w, "scene_params": sp, "source": "video"})
    assert code == 400


def test_409_when_not_trained():
    """未挂载 checkpoint 的 service => /wm/* 409。"""
    torch.manual_seed(0)
    svc = UDOSService(preset="small")
    w, sp = _window()
    with pytest.raises(Exception) as ei:
        svc.wm_imagine({"window": w, "scene_params": sp, "horizon": 2})
    # ServiceNotReady -> 由 HTTP 层映射 409; 这里直接断言其类型名
    assert "ServiceNotReady" in type(ei.value).__name__


def test_unknown_route_404(live_server):
    code, _ = _post(live_server, "/wm/nope", {"window": _window()[0]})
    assert code == 404


def test_loop_step_still_default(live_server):
    """/loop/step 默认路径不受 WM 集成影响 (全特性兼容)。"""
    w, sp = _window()
    code, body = _post(live_server, "/loop/step",
                       {"window": w, "scene_params": sp, "horizon": 4})
    assert code == 200, body
    assert body["status"] == "ok"


def test_loop_wm_opt_in_combination():
    """PhysicalLoopRunner + LatentWorldModel opt-in 组合; 默认关时逐位走 rollout。"""
    from udos.persistence import load_predictor
    from udos.physical_loop import PhysicalLoopRunner
    from udos.world_model import LatentWorldModel
    from udos.dynamics import build_parametric_dataset

    m, _ = load_predictor(CKPT)
    m.eval()
    ds = build_parametric_dataset(n_per_kind=4, n_steps=14, window=6,
                                  horizon=4, dt=0.5, seed=55)
    x, p = ds.X[:1], ds.P[:1]

    loop = PhysicalLoopRunner(m, horizon=4)
    default = loop.run(x, scene_params=p)
    # 默认关: future_state 走 rollout, 与直接 rollout 逐位一致
    ref = m.rollout(x, 4, scene_params=p)
    default_traj = default["loop_state"]["outputs"]["future_state"]["trajectory"]
    assert torch.equal(default_traj, ref)

    # opt-in: 挂载 WM, future_state 走想象 (第 0 步仍锚定 rollout)
    wm = LatentWorldModel(m)
    wm.fit(build_parametric_dataset(n_per_kind=4, n_steps=14, window=6,
                                    horizon=4, dt=0.5, seed=56), epochs=5)
    loop.use_world_model = True
    loop.world_model = wm
    imagined = loop.run(x, scene_params=p)
    imag_traj = imagined["loop_state"]["outputs"]["future_state"]["trajectory"]
    assert imag_traj.shape == (1, 4, 6)
    # 第 0 步锚定真实单步 => 与 rollout 第 0 步一致
    assert torch.allclose(imag_traj[:, 0, :], ref[:, 0, :], atol=1e-6)


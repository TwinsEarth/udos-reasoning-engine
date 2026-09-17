"""v2.5.1 回归: POST /rollback (回滚到上一已加载 checkpoint)。

覆盖:
- 加载两个 checkpoint 后回滚 -> 恢复前一个
- 无历史 (栈长 < 2) -> ServiceNotReady (409)
- 回滚后预测一致
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from udos import __version__  # noqa: E402
from udos.server import ServiceNotReady  # noqa: E402


def test_version_bumped():
    assert __version__ == "5.5.5"


def test_rollback_no_history_409(tmp_path, monkeypatch):
    """栈长 < 2 时回滚报 ServiceNotReady (对应 409)。"""
    monkeypatch.chdir(tmp_path)
    from udos.server import UDOSService
    s = UDOSService(preset="small")
    s.train({"epochs": 1, "n_per_kind": 6, "horizon": 3})
    # 训练后栈为空 (train 不走 load), 回滚应 409
    try:
        s.rollback({})
        assert False, "无历史应报 ServiceNotReady"
    except ServiceNotReady:
        pass


def test_rollback_after_two_loads(tmp_path, monkeypatch):
    """加载两个 checkpoint 后回滚 -> 恢复前一个。"""
    monkeypatch.chdir(tmp_path)
    import os
    from udos.server import UDOSService

    ck_dir = str(tmp_path / "checkpoints")
    # 训练并保存两个 checkpoint (save 名称直接用作文件名)
    s = UDOSService(preset="small", checkpoints_dir=ck_dir)
    s.train({"epochs": 1, "n_per_kind": 6, "horizon": 3})
    os.makedirs(ck_dir, exist_ok=True)
    s.save({"name": "ckptA"})
    s.save({"name": "ckptB"})

    # 重新构造 service 并 load (save 不进栈, load 才进栈)
    s2 = UDOSService(preset="small", checkpoints_dir=ck_dir)
    assert len(s2._load_stack) == 0
    s2.load({"name": "ckptA"})
    s2.load({"name": "ckptB"})
    assert len(s2._load_stack) == 2

    # 回滚 -> 恢复 A
    out = s2.rollback({})
    assert out["status"] == "rolled_back"
    assert "ckptA" in out["restored_path"]
    assert len(s2._load_stack) == 1

    # 再回滚 -> 无历史 -> 409
    try:
        s2.rollback({})
        assert False
    except ServiceNotReady:
        pass


def test_rollback_after_pretrained_startup(tmp_path, monkeypatch):
    """启动时 --checkpoint 预加载占栈底, load 另一个后可回滚。"""
    monkeypatch.chdir(tmp_path)
    import os
    from udos.server import UDOSService

    ck_dir = str(tmp_path / "checkpoints")
    # 先训练并保存
    s = UDOSService(preset="small", checkpoints_dir=ck_dir)
    s.train({"epochs": 1, "n_per_kind": 6, "horizon": 3})
    os.makedirs(ck_dir, exist_ok=True)
    s.save({"name": "preload_base"})

    # 以预加载启动 (栈底=preload_base)
    s2 = UDOSService(preset="small", checkpoints_dir=ck_dir,
                      checkpoint=str(tmp_path / "checkpoints" / "preload_base.pt"))
    assert len(s2._load_stack) == 1
    # 再 load 同一个 (栈长=2)
    s2.load({"name": "preload_base"})
    assert len(s2._load_stack) == 2
    # 回滚 -> 恢复栈底
    out = s2.rollback({})
    assert out["status"] == "rolled_back"
    assert len(s2._load_stack) == 1
    # 再回滚 -> 409
    try:
        s2.rollback({})
        assert False
    except ServiceNotReady:
        pass


def test_rollback_prediction_consistent(tmp_path, monkeypatch):
    """回滚后预测与该 checkpoint 单独加载一致。"""
    monkeypatch.chdir(tmp_path)
    import os
    from udos.server import UDOSService
    from udos.dynamics import build_parametric_dataset
    ck_dir = str(tmp_path / "checkpoints")
    s = UDOSService(preset="small", checkpoints_dir=ck_dir)
    s.train({"epochs": 1, "n_per_kind": 6, "horizon": 3})
    os.makedirs(ck_dir, exist_ok=True)
    s.save({"name": "rollback_test"})

    ckpt_path = str(tmp_path / "checkpoints" / "rollback_test.pt")
    s2 = UDOSService(preset="small", checkpoints_dir=ck_dir, checkpoint=ckpt_path)
    s2.load({"name": "rollback_test"})  # 栈长=2
    ds = build_parametric_dataset(n_per_kind=6, n_steps=14, window=6,
                                  horizon=3, dt=0.5, seed=77)
    before = s2.predict({"window": ds.X[:2].tolist(),
                         "scene_params": ds.P[:2].tolist()})
    s2.rollback({})  # 回滚到栈底
    after = s2.predict({"window": ds.X[:2].tolist(),
                         "scene_params": ds.P[:2].tolist()})
    # 同一权重, 预测应一致
    assert before["prediction"] == after["prediction"]

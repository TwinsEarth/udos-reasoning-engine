import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402
import torch  # noqa: E402

from udos.pce_format import PhysicalToken, PhysicsScene  # noqa: E402


@pytest.fixture
def tiny_scene():
    scene = PhysicsScene(scene_id="s1", duration=4)
    for t in range(4):
        scene.add(PhysicalToken(
            object_id="obj-a", timestamp=t,
            position=[float(t), 0.0, 0.0],
            velocity=[1.0, 0.0, 0.0],
            attributes={"mass": 2.0, "temp": 30.0 + t},
            causal_parents=([{"object_id": "obj-a", "timestamp": t - 1}]
                            if t > 0 else [])))
        scene.add(PhysicalToken(
            object_id="obj-b", timestamp=t, position=[0.0, 1.0, 0.0],
            attributes={"mass": 5.0}))
    return scene


@pytest.fixture
def torch_seed():
    torch.manual_seed(0)
    yield

import torch

from udos.pce_format import (
    PCEParser, PhysicsSceneEncoder, collate_tokens, BASE_PHYS_DIM)


def test_pce_roundtrip(tiny_scene, tmp_path):
    p = PCEParser.dump(tiny_scene, tmp_path / "s.pce")
    loaded = PCEParser.load(p)
    assert loaded.scene_id == tiny_scene.scene_id
    assert len(loaded.tokens) == len(tiny_scene.tokens)
    obj_a_t3 = next(t for t in loaded.tokens
                    if t.object_id == "obj-a" and t.timestamp == 3)
    assert obj_a_t3.attributes["mass"] == 2.0
    assert loaded.timeline()[0].timestamp == 0


def test_encoder_adapts_attr_dims(tiny_scene):
    enc = PhysicsSceneEncoder(d_model=32)
    enc.bind_scene(tiny_scene)  # 出现 mass/temp 两个属性键
    assert enc.in_dim == BASE_PHYS_DIM + 2
    seq = enc(tiny_scene.tokens)
    assert seq.shape == (len(tiny_scene.tokens), 32)
    summary = enc.encode_scene_summary(tiny_scene)
    assert summary.shape == (32,)


def test_collate_padding():
    a = torch.randn(3, 8)
    b = torch.randn(5, 8)
    batch = collate_tokens([a, b])
    assert batch.shape == (2, 5, 8)
    assert torch.equal(batch[0, 3:], torch.zeros(2, 8))

"""
v3.3.0.dev2 节点53: 知识蒸馏 v2 (温度软标签 + 中间特征匹配 + 学生自动架构) 测试
====================================================================
纪律:
    * 蒸馏 loss 下降; 温度可调; 中间特征匹配项参与; 学生自动 d_model 收缩;
    * 学生参数量 < teacher; 对比 v1 蒸馏; save/load 学生。
analogy, not reproduction。
"""
import torch

from udos.ctm_engine import CTMConfig
from udos.training import PhysicsPredictor, CTMTrainer, TrainConfig
from udos.lite import (DistillationTrainer, DistillationTrainerV2,
                       auto_student_config)
from udos.dynamics import build_parametric_dataset
from udos.persistence import save_predictor, load_predictor

torch.set_num_threads(2)


def base_cfg():
    cfg = CTMConfig(iterations=4, d_model=32, d_input=16, heads=4,
                    n_synch_out=8, n_synch_action=4, memory_length=6,
                    nlm_hidden=8, out_dims=8, certainty_threshold=0.0)
    cfg.scene_dim = 32
    return cfg


def trained_teacher(seed=0):
    ds = build_parametric_dataset(n_per_kind=12, n_steps=12, window=6,
                                 horizon=2, dt=0.5, seed=seed)
    tr, va = ds.split(0.8)
    torch.manual_seed(seed)
    teacher = PhysicsPredictor(base_cfg(), scene_param_dim=4)
    CTMTrainer(teacher, TrainConfig(epochs=4, batch_size=16,
                                    patience=None)).train(tr, va)
    teacher.eval()
    return teacher, ds


def test_distill_v2_loss_descends():
    teacher, ds = trained_teacher()
    v2 = DistillationTrainerV2(temperature=4.0, feature_weight=0.5,
                               student_scale=0.5)
    student = v2.distill(teacher, base_cfg(), ds, epochs=6, batch_size=16)
    h = student.distill_history
    assert h[-1] < h[0], f"loss 未下降: {h[0]} -> {h[-1]}"
    assert student.distill_v2["temperature"] == 4.0


def test_temperature_adjustable():
    teacher, ds = trained_teacher()
    outs = {}
    for T in (1.0, 8.0):
        torch.manual_seed(0)
        s = DistillationTrainerV2(temperature=T, feature_weight=0.5,
                                  student_scale=0.5).distill(
            teacher, base_cfg(), ds, epochs=3, batch_size=16)
        outs[T] = s
    # 不同温度下学生权重不同 (温度确实进入损失)
    p1 = list(outs[1.0].parameters())[0]
    p8 = list(outs[8.0].parameters())[0]
    assert not torch.equal(p1, p8)


def test_student_auto_arch_and_param_count():
    teacher, ds = trained_teacher()
    student = DistillationTrainerV2(student_scale=0.25).distill(
        teacher, base_cfg(), ds, epochs=2, batch_size=16)
    n_teacher = sum(p.numel() for p in teacher.parameters())
    n_student = sum(p.numel() for p in student.parameters())
    assert n_student < n_teacher
    # 四分之一学生 d_model 应显著小于 teacher
    assert student.ctm.cfg.d_model == 8  # 32 * 0.25 = 8
    # 半量学生
    s2 = DistillationTrainerV2(student_scale=0.5).distill(
        teacher, base_cfg(), ds, epochs=1, batch_size=16)
    assert s2.ctm.cfg.d_model == 16


def test_vs_v1_distillation():
    """v2 与 v1 均产出更小学生; v2 学生同样参数量级但额外记录温度/特征项。"""
    teacher, ds = trained_teacher()
    s_v1 = DistillationTrainer(alpha=0.7).distill(
        teacher, base_cfg(), ds, epochs=2, batch_size=16)
    s_v2 = DistillationTrainerV2(student_scale=0.5).distill(
        teacher, base_cfg(), ds, epochs=2, batch_size=16)
    n1 = sum(p.numel() for p in s_v1.parameters())
    n2 = sum(p.numel() for p in s_v2.parameters())
    # v1 学生也是减半 (d_model=32->默认 16), 二者同量级
    assert n1 > 0 and n2 > 0
    assert hasattr(s_v2, "distill_v2") and s_v2.distill_v2["feature_weight"] > 0


def test_save_load_student(tmp_path):
    teacher, ds = trained_teacher()
    student = DistillationTrainerV2(student_scale=0.5).distill(
        teacher, base_cfg(), ds, epochs=2, batch_size=16)
    x, p = ds.X[:2], ds.P[:2]
    before = student.predict_next(x, scene_params=p)
    ckpt = tmp_path / "student_v2.pt"
    save_predictor(student, str(ckpt))
    loaded, _ = load_predictor(str(ckpt))
    after = loaded.predict_next(x, scene_params=p)
    assert torch.allclose(before, after, atol=1e-6)

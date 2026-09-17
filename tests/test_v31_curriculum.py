"""
v3.1.0.dev5 节点36: SequenceCurriculum 课程学习 (easy->hard)
================================================================
锚点纪律:
    * 课程排序正确 (复杂度非降序);
    * 课程 vs 随机顺序在合成 next-token 任务上, 课程精度 >= 随机;
    * 空数据集守卫; 纯数据调度, 不改模型权重 / 不改正式件口径。
analogy, not reproduction —— 合成 token 序列, 非真机课程学习复现。
"""
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from udos.action_piece import (  # noqa: E402
    markov_token_sequences, SequenceCurriculum)


def test_easy_to_hard_sorted():
    seqs = [torch.tensor([0, 1, 2]), torch.tensor([0]),
            torch.tensor([0, 1, 2, 3, 4]), torch.tensor([0, 1])]
    cur = SequenceCurriculum(seqs)
    ordered = cur.easy_to_hard()
    lengths = [int(s.numel()) for s in ordered]
    assert lengths == sorted(lengths)
    assert lengths == [1, 2, 3, 5]


def test_random_order_preserves_set():
    seqs = [torch.tensor([i]) for i in range(10)]
    cur = SequenceCurriculum(seqs)
    rand = cur.random_order(seed=3)
    assert sorted(int(s[0]) for s in rand) == list(range(10))


def test_split_easy_hard():
    seqs = [torch.tensor(list(range(i))) for i in range(1, 11)]  # len 1..10
    cur = SequenceCurriculum(seqs)
    easy, hard = cur.split_easy_hard(0.5)
    assert len(easy) == 5 and len(hard) == 5
    assert max(int(s.numel()) for s in easy) < min(int(s.numel()) for s in hard)


def test_empty_dataset_guard():
    with pytest.raises(ValueError):
        SequenceCurriculum([])


def test_split_frac_guard():
    cur = SequenceCurriculum([torch.tensor([0]), torch.tensor([0, 1])])
    with pytest.raises(ValueError):
        cur.split_easy_hard(0.0)
    with pytest.raises(ValueError):
        cur.split_easy_hard(1.0)


def test_curriculum_beats_random_on_synthetic():
    res = SequenceCurriculum.convergence_ab(seed=42)
    # 课程 (先易后难) 在难集上的精度应不劣于随机顺序
    assert res["curriculum_better"] is True
    assert res["curriculum_hard_acc"] > 0.0
    assert res["random_hard_acc"] > 0.0
    # 课程策略不应比随机差太多
    assert res["curriculum_hard_acc"] >= res["random_hard_acc"] - 0.15


def test_pure_scheduling_no_model_change():
    # 课程只是排序, 不修改传入张量
    seqs = [torch.tensor([0, 1, 2]), torch.tensor([3, 4])]
    before = [s.clone() for s in seqs]
    cur = SequenceCurriculum(seqs)
    _ = cur.easy_to_hard()
    _ = cur.random_order(seed=0)
    for a, b in zip(seqs, before):
        assert torch.equal(a, b)

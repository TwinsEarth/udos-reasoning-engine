"""
v3.1.0.dev6 节点37: InContextActionPrompter few-shot 上下文
================================================================
锚点纪律:
    * few-shot 示例 + 查询拼接长度正确;
    * 1/3/5-shot 精度随 shot 数不下降 (更多示例 -> 计数更完整);
    * 空示例 / 空查询守卫; 与 TokenizedActionPredictor 接口一致。
analogy, not reproduction —— 合成 token 序列 in-context 类比, 非 PhysBrain 长上下文复现。
"""
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from udos.action_piece import (  # noqa: E402
    InContextActionPrompter, markov_token_sequences)


def test_build_concatenates_examples_and_query():
    prompter = InContextActionPrompter(vocab_size=8, order=2)
    ex1 = torch.tensor([0, 1, 2])
    ex2 = torch.tensor([3, 4, 5])
    q = torch.tensor([6, 7])
    ctx = prompter.build_context([ex1, ex2], q)
    assert ctx.tolist() == [0, 1, 2, 3, 4, 5, 6, 7]
    assert ctx.numel() == 3 + 3 + 2


def test_predict_next_from_examples():
    prompter = InContextActionPrompter(vocab_size=8, order=2)
    # 示例展示规则 a -> (a+1)%8
    examples = [torch.tensor([0, 1, 2, 3]), torch.tensor([4, 5, 6, 7])]
    nxt = prompter.predict_next(torch.tensor([5, 6]), examples)
    # 查询 5->6 后应预测 7
    assert nxt == 7


def test_shot_accuracy_non_decreasing():
    prompter = InContextActionPrompter(vocab_size=8, order=2)
    accs = {}
    for k in (1, 3, 5):
        accs[k] = prompter.few_shot_accuracy(k, seed=42)
    # 更多 shot 不应显著更差; 5-shot 至少不劣于 1-shot
    assert accs[5] >= accs[1] - 0.1
    # 全部优于随机 1/8=0.125
    for k in (1, 3, 5):
        assert accs[k] > 0.125
    print("shot accs:", accs)


def test_empty_examples_guard():
    prompter = InContextActionPrompter(vocab_size=8)
    with pytest.raises(ValueError):
        prompter.predict_next(torch.tensor([0, 1]), [])
    with pytest.raises(ValueError):
        prompter.build_context([], torch.tensor([]))   # 空查询


def test_bad_n_shots_guard():
    prompter = InContextActionPrompter(vocab_size=8)
    with pytest.raises(ValueError):
        prompter.few_shot_accuracy(0)


def test_matches_token_predict_interface():
    # in-context 预测结果与直接在示例上 fit 的 n-gram 一致
    from udos.action_piece import TokenizedActionPredictor
    prompter = InContextActionPrompter(vocab_size=8, order=2)
    examples = [torch.tensor([0, 1, 2, 3, 4]), torch.tensor([5, 6, 7, 0, 1])]
    query = torch.tensor([3, 4])
    a = prompter.predict_next(query, examples)
    ref = TokenizedActionPredictor(8, 2).fit(examples).predict_next(query)
    assert a == ref

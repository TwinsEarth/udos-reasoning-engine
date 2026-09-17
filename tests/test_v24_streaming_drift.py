"""v2.4.14 回归: StreamingDriftDetector — 流式==批处理、窗口重置、漂移检测、与离线同语义。"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from udos import __version__  # noqa: E402
from udos.ood import StreamingDriftDetector, DistributionDriftDetector  # noqa: E402


def test_version_bumped():
    assert __version__ == "5.5.5"


def _ref_and_data():
    torch.manual_seed(0)
    ref = torch.randn(200, 4)
    return ref


def test_streaming_equals_batch():
    ref = _ref_and_data()
    det = StreamingDriftDetector(window=50).fit(ref)
    # 一批新样本逐行 update
    torch.manual_seed(1)
    stream = torch.randn(30, 4)
    for i in range(30):
        det.update(stream[i])
    # 流式窗口统计 == 对这 30 行批处理统计
    assert det.n_window == 30
    assert torch.allclose(det.window_mean(), stream.mean(dim=0), atol=1e-6)
    assert torch.allclose(det.window_var(),
                          stream.var(dim=0, unbiased=False), atol=1e-6)


def test_window_bounded_and_evicts_oldest():
    det = StreamingDriftDetector(window=5).fit(_ref_and_data())
    for i in range(10):
        det.update(torch.full((4,), float(i)))
    assert det.n_window == 5                       # 窗口封顶
    # 窗口内应为最后 5 条 (5..9)
    expected = torch.arange(5, 10, dtype=torch.float32).unsqueeze(1) \
        .expand(5, 4).mean(dim=0)
    assert torch.allclose(det.window_mean(), expected, atol=1e-6)


def test_reset_clears_window_keeps_reference():
    ref = _ref_and_data()
    det = StreamingDriftDetector(window=10).fit(ref)
    det.update(torch.randn(4))
    assert det.n_window == 1
    det.reset()
    assert det.n_window == 0
    assert det.window_mean() is None
    # 参考分布仍在 (reset 不清参考)
    assert det.fitted
    assert det.score(torch.randn(3, 4)).numel() == 3


def test_drift_detection_and_offline_parity():
    ref = _ref_and_data()
    det = StreamingDriftDetector(window=64).fit(ref)
    off = DistributionDriftDetector().fit(ref)
    torch.manual_seed(2)
    id_x = torch.randn(20, 4)
    ood_x = torch.randn(20, 4) + 5.0            # 显著偏移
    # score 与离线检测器逐位一致 (继承同语义)
    assert torch.allclose(det.score(id_x), off.score(id_x), atol=1e-5)
    # OOD 样本分高、ID 样本分低
    assert det.score(ood_x).mean() > det.score(id_x).mean()
    # is_ood: 偏移样本 OOD 率更高
    assert det.is_ood(ood_x).float().mean() >= det.is_ood(id_x).float().mean()


def test_window_drift_score():
    ref = _ref_and_data()
    det = StreamingDriftDetector(window=32).fit(ref)
    assert det.window_drift_score() != det.window_drift_score()  # 空窗口 nan
    for _ in range(20):
        det.update(torch.randn(4))               # ID 样本
    low = det.window_drift_score()
    det.reset()
    for _ in range(20):
        det.update(torch.randn(4) + 5.0)         # OOD 样本
    high = det.window_drift_score()
    assert high > low

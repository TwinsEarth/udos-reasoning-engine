"""M3 探针：conformal 按 α 真分水平 + 覆盖率达标 + 参数蒙特卡洛扇形。

关键反例（旧版 conformal_by_alpha 的 bug）：名义 80/90/95 带宽与经验覆盖完全相同。
split conformal 在交换性下对**任意模型**（含未训练恒等模型）保证边际覆盖，
因此本测试无需训练即可严格判定 α 是否真的生效。
"""
import torch

from udos7 import WorldModelCore
from udos7.dynamics import three_way_splits
from udos7.uncertainty import (ConformalCalibrator, MonteCarloParamFan,
                               empirical_coverage)

ALPHAS = (0.2, 0.1, 0.05)
NOMINAL = {0.2: 0.80, 0.1: 0.90, 0.05: 0.95}


def _model_and_splits():
    torch.manual_seed(0)
    model = WorldModelCore(window=6, hidden=64, n_layers=1)
    model.eval()
    splits = three_way_splits(n_traj_per_kind=16)
    return model, splits


def test_bandwidths_distinct_and_ordered_by_alpha():
    model, splits = _model_and_splits()
    cal = ConformalCalibrator(horizon=4).fit(
        model, splits["calib"], use_explicit=True, alphas=ALPHAS)
    q = {a: cal.bands[a].q_marginal for a in ALPHAS}
    # 名义覆盖越高(α 越小)，带宽必须越大，且严格不同
    assert q[0.2] < q[0.1] < q[0.05]
    assert cal.widths_depend_on_alpha()
    # 逐步带宽也必须随 α 变化
    assert not torch.allclose(cal.bands[0.2].q_per_step,
                              cal.bands[0.05].q_per_step)


def test_empirical_coverage_matches_nominal_oracle():
    model, splits = _model_and_splits()
    cal = ConformalCalibrator(horizon=4).fit(
        model, splits["calib"], use_explicit=True, alphas=ALPHAS)
    test = splits["test"]
    for a in ALPHAS:
        iv = cal.predict_interval(model, test.X, 4, a, explicit=test.P)
        cov = empirical_coverage(iv["lower"], iv["upper"], test.Y)["marginal"]
        # 交换性保证 + 窗口相关的有效样本折损：容差 0.05
        assert abs(cov - NOMINAL[a]) <= 0.05, (
            f"α={a} 名义 {NOMINAL[a]} 实测 {cov}，conformal 未按水平分带")


def test_coverage_higher_alpha_band_covers_more():
    model, splits = _model_and_splits()
    cal = ConformalCalibrator(horizon=4).fit(
        model, splits["calib"], use_explicit=True, alphas=ALPHAS)
    test = splits["test"]
    covs = {}
    for a in ALPHAS:
        iv = cal.predict_interval(model, test.X, 4, a, explicit=test.P)
        covs[a] = empirical_coverage(iv["lower"], iv["upper"], test.Y)["marginal"]
    # 95% 带必须比 80% 带覆盖更多
    assert covs[0.05] > covs[0.2]


def test_monte_carlo_param_fan_ordering_and_band():
    # 参数扇形要求参数真正影响预测 => 必须用训练过的模型（未训练残差头为零，
    # 参数不影响输出，M 次 draw 完全相同，扇形退化）。
    from udos7.train import TrainConfig, fit
    torch.manual_seed(0)
    model = WorldModelCore(window=6, hidden=64, n_layers=1)
    splits = three_way_splits(n_traj_per_kind=12)
    fit(model, splits["train"], splits["val"],
        TrainConfig(epochs=25, batch=64, lr=3e-3, patience=25, seed=0))
    model.eval()
    fan = MonteCarloParamFan(draws=32).fit(model, splits["calib"])
    fan.calibrate_inflation(model, splits["calib"], 4, nominal=0.8, max_n=96)
    f = fan.sample(model, splits["test"].X[:96], 4)
    assert torch.all(f.p10 <= f.p50 + 1e-6)
    assert torch.all(f.p50 <= f.p90 + 1e-6)
    # conformal 膨胀后扇形必须有非零带宽
    assert float((f.p90 - f.p10).mean()) > 1e-3
    truth = splits["test"].Y[:96]
    inside = ((truth >= f.p10) & (truth <= f.p90)).float().mean().item()
    # 目标名义 80%，乘法 conformal + 窗口相关折损，容差 [0.70,0.95]
    assert 0.70 <= inside <= 0.95, f"膨胀后扇形覆盖 {inside} 偏离名义 0.80"

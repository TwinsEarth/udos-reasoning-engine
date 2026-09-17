"""
v2.7.0.dev5 实验注册与多种子 sweep 治理 (udos.experiment.ExperimentRegistry) 测试
============================================================================
锚点纪律:
    * 注册/查询/列表正确;
    * 多种子聚合: 同名不同 seed 的 metrics 计算 mean/std/best;
    * JSON 原子持久化与重载后逐字段一致;
    * 空注册表守卫 (list 空 / sweep_report 空 / get 未知名 KeyError);
    * 同 (name, seed) 重复注册幂等覆盖, 不产生重复记录。
"""
import json

import pytest

from udos import __version__
from udos.experiment import ExperimentRegistry


def test_version():
    assert __version__ == "5.5.5"


def test_register_list_get():
    reg = ExperimentRegistry()
    reg.register("exp_a", config={"k": 1}, metrics={"mse": 0.5}, seed=0,
                 artifacts=["a.pt"])
    reg.register("exp_a", config={"k": 1}, metrics={"mse": 0.6}, seed=1)
    reg.register("exp_b", config={"k": 2}, metrics={"mse": 0.7}, seed=0)
    assert reg.list() == ["exp_a", "exp_b"]
    assert len(reg) == 3
    got = reg.get("exp_a")
    assert [r["seed"] for r in got] == [0, 1]
    assert got[0]["metrics"]["mse"] == 0.5
    assert got[0]["artifacts"] == ["a.pt"]
    assert got[1]["artifacts"] == []


def test_get_unknown_raises():
    reg = ExperimentRegistry()
    with pytest.raises(KeyError):
        reg.get("nope")


def test_sweep_aggregate_stats():
    reg = ExperimentRegistry()
    # exp_a: 3 个 seed 的 mse; 越小越好 => best = min
    for s, mse in enumerate([0.4, 0.6, 0.5]):
        reg.register("exp_a", {}, {"mse": mse}, seed=s)
    rep = reg.sweep_report()
    assert rep["n_names"] == 1
    a = rep["experiments"]["exp_a"]
    assert a["seeds"] == [0, 1, 2]
    st = a["metrics"]["mse"]
    assert st["n_seeds"] == 3
    assert st["mean"] == pytest.approx((0.4 + 0.6 + 0.5) / 3, abs=1e-6)
    assert st["best"] == pytest.approx(0.4, abs=1e-9)     # min, 越小越好
    assert st["std"] == pytest.approx(
        ((((0.4 - 0.5) ** 2 + (0.6 - 0.5) ** 2 + (0.5 - 0.5) ** 2) / 3) ** 0.5),
        abs=1e-6)
    # 越大越好的 key (coverage) => best = max
    reg.register("exp_cov", {}, {"coverage": 0.8, "x": 1.0}, seed=0)
    reg.register("exp_cov", {}, {"coverage": 0.9, "x": 2.0}, seed=1)
    rep2 = reg.sweep_report(["exp_cov"])
    cov = rep2["experiments"]["exp_cov"]["metrics"]["coverage"]
    assert cov["best"] == pytest.approx(0.9, abs=1e-9)


def test_sweep_subset_names():
    reg = ExperimentRegistry()
    reg.register("a", {}, {"mse": 1.0}, seed=0)
    reg.register("b", {}, {"mse": 2.0}, seed=0)
    rep = reg.sweep_report(["a"])
    assert rep["n_names"] == 1
    assert "a" in rep["experiments"] and "b" not in rep["experiments"]


def test_empty_registry_guards():
    reg = ExperimentRegistry()
    assert reg.list() == []
    assert len(reg) == 0
    rep = reg.sweep_report()
    assert rep["n_names"] == 0
    assert rep["experiments"] == {}


def test_duplicate_register_idempotent_overwrite():
    reg = ExperimentRegistry()
    reg.register("a", {"v": 1}, {"mse": 0.5}, seed=0)
    reg.register("a", {"v": 2}, {"mse": 0.9}, seed=0)   # 同 name+seed
    assert len(reg) == 1
    got = reg.get("a")
    assert len(got) == 1
    assert got[0]["metrics"]["mse"] == 0.9               # 覆盖后的值
    assert got[0]["config"]["v"] == 2


def test_save_load_roundtrip(tmp_path):
    reg = ExperimentRegistry()
    reg.register("train", {"lr": 3e-3}, {"mse": 0.42, "ece": 0.03}, seed=0,
                 artifacts=["ckpt.pt"])
    reg.register("train", {"lr": 3e-3}, {"mse": 0.40, "ece": 0.02}, seed=1)
    path = str(tmp_path / "sub" / "reg.json")
    out = reg.save(path)
    assert out == path
    assert __import__("os").path.isfile(path)

    reg2 = ExperimentRegistry(path)
    reg2.load(path)
    assert reg2.list() == ["train"]
    recs = reg2.get("train")
    assert [r["seed"] for r in recs] == [0, 1]
    assert recs[0]["metrics"]["mse"] == 0.42
    assert recs[1]["metrics"]["mse"] == 0.40
    assert recs[0]["artifacts"] == ["ckpt.pt"]
    # 重载后 sweep 仍可算
    rep = reg2.sweep_report()
    assert rep["experiments"]["train"]["metrics"]["mse"]["n_seeds"] == 2


def test_load_missing_file_is_empty(tmp_path):
    reg = ExperimentRegistry(str(tmp_path / "nope.json"))
    reg.load()          # 文件不存在 -> 空注册表, 不报错
    assert reg.list() == []


def test_register_validation():
    reg = ExperimentRegistry()
    with pytest.raises(ValueError):
        reg.register("", {}, {"mse": 0.1}, seed=0)
    with pytest.raises(ValueError):
        reg.register("x", "notadict", {"mse": 0.1}, seed=0)

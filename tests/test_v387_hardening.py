"""v3.8.7 hardening FIX-301 回归测试
=====================================
缺陷: 新端点 (twin/wm/icm) 直接 int(body.get(...))/float(body.get(...)),
当 JSON 值为 null (Python None) 时, int(None) 抛 TypeError,
不被 do_POST 的 except ValueError 捕获, 落入 except Exception -> 500。

修复:
1. do_POST 异常处理从 except ValueError 扩展为 except (ValueError, TypeError)
2. UDOSService 新增 _coerce_int/_coerce_float 辅助方法, 新端点使用之
3. neural_step 增加 candidate_actions 每项须为 dict 的校验
"""
import pytest
from udos.server import UDOSService


@pytest.fixture(scope="module")
def svc():
    return UDOSService(preset="small", checkpoint="checkpoints/predictor_v3.8.6.pt")


# ---- null 输入不崩溃 (视为缺省) ---- #

class TestFIX301NullInput:
    def test_twin_scene_null_n_agents(self, svc):
        out = svc.twin_scene({"n_agents": None, "n_obstacles": 1,
                              "seed": 1, "bounds": 5.0})
        assert out["status"] == "ok"

    def test_twin_scene_null_bounds(self, svc):
        out = svc.twin_scene({"n_agents": 2, "n_obstacles": 1,
                              "seed": 1, "bounds": None})
        assert out["status"] == "ok"

    def test_twin_step_null_dt(self, svc):
        svc.twin_scene({"n_agents": 2, "n_obstacles": 1, "seed": 1,
                        "bounds": 5.0})
        out = svc.twin_step({"dt": None})
        assert out["status"] == "ok"

    def test_wm_imagine_null_horizon(self, svc):
        out = svc.wm_imagine({"window": [[0.0] * 6] * 6, "horizon": None})
        assert out["status"] == "ok"

    def test_wm_conservation_null_mass(self, svc):
        out = svc.wm_conservation(
            {"window": [[0.0] * 6] * 6, "horizon": 2,
             "source": "real", "mass": None})
        assert out["status"] == "ok"

    def test_wm_conservation_null_spring_k(self, svc):
        out = svc.wm_conservation(
            {"window": [[0.0] * 6] * 6, "horizon": 2,
             "source": "real", "spring_k": None})
        assert out["status"] == "ok"

    def test_icm_predict_null_k(self, svc):
        svc.icm_demo_register(
            {"input_window": [[0.0] * 6] * 6, "result": [0.0] * 6})
        out = svc.icm_predict({"window": [[0.0] * 6] * 6, "k": None})
        assert out["status"] == "ok"


# ---- 字符串非法输入仍 400 (ValueError) ---- #

class TestFIX301BadString:
    def test_twin_scene_bad_n_agents(self, svc):
        with pytest.raises(ValueError):
            svc.twin_scene({"n_agents": "abc"})

    def test_twin_scene_bad_bounds(self, svc):
        with pytest.raises(ValueError):
            svc.twin_scene({"bounds": "xyz"})

    def test_twin_step_bad_dt(self, svc):
        with pytest.raises(ValueError):
            svc.twin_step({"dt": "abc"})

    def test_wm_imagine_bad_horizon(self, svc):
        with pytest.raises(ValueError):
            svc.wm_imagine({"window": [[0.0] * 6] * 6, "horizon": "abc"})

    def test_wm_conservation_bad_mass(self, svc):
        with pytest.raises(ValueError):
            svc.wm_conservation(
                {"window": [[0.0] * 6] * 6, "horizon": 2,
                 "source": "real", "mass": "xyz"})

    def test_icm_predict_bad_k(self, svc):
        with pytest.raises(ValueError):
            svc.icm_predict({"window": [[0.0] * 6] * 6, "k": "abc"})


# ---- candidate_actions 每项须为 dict ---- #

class TestFIX301CandidateActions:
    def test_non_dict_item_raises(self, svc):
        with pytest.raises(ValueError, match="每项需为 dict"):
            svc.neural_step(
                {"window": [[0.0] * 6] * 6, "candidate_actions": [1, 2, 3]})

    def test_dict_items_ok(self, svc):
        out = svc.neural_step(
            {"window": [[0.0] * 6] * 6,
             "candidate_actions": [{"candidate_state": [0.0] * 6}]})
        assert out["status"] == "ok"


# ---- HTTP 层 TypeError -> 400 兜底 ---- #

class TestFIX301HTTPTypeError:
    """do_POST 现在捕获 (ValueError, TypeError) -> 400。"""

    def test_helper_int_none_uses_default(self):
        assert UDOSService._coerce_int(None, 5, "x") == 5

    def test_helper_int_bad_string(self):
        with pytest.raises(ValueError, match="无法解析为整数"):
            UDOSService._coerce_int("abc", 5, "x")

    def test_helper_float_none_uses_default(self):
        assert UDOSService._coerce_float(None, 1.0, "x") == 1.0

    def test_helper_float_bad_string(self):
        with pytest.raises(ValueError, match="无法解析为浮点数"):
            UDOSService._coerce_float("xyz", 1.0, "x")


# ---- 工作线3: 关键路径 caplog 断言 ---- #

class TestLoggingKeyPaths:
    def test_wm_lazy_init_logs(self, svc, caplog):
        """WM 首次懒构造应记 INFO 日志。"""
        import logging
        caplog.set_level(logging.INFO, logger="udos")
        svc._wm = None
        svc.wm_imagine({"window": [[0.0] * 6] * 6, "horizon": 2})
        assert any("WM latent model" in r.message for r in caplog.records)

    def test_neural_lazy_init_logs(self, svc, caplog):
        """神经控制器首次懒构造应记 INFO 日志。"""
        import logging
        caplog.set_level(logging.INFO, logger="udos")
        svc._neural = None
        svc.neural_step({"window": [[0.0] * 6] * 6})
        assert any("HierarchicalController" in r.message
                   for r in caplog.records)

    def test_twin_scene_create_logs(self, svc, caplog):
        """孪生场景创建应记 INFO 日志。"""
        import logging
        caplog.set_level(logging.INFO, logger="udos")
        svc.twin_scene({"n_agents": 2, "n_obstacles": 1, "seed": 42,
                        "bounds": 5.0})
        assert any("DigitalTwinScene" in r.message for r in caplog.records)

    def test_icm_register_logs(self, svc, caplog):
        """ICM demo 注册应记 INFO 日志。"""
        import logging
        caplog.set_level(logging.INFO, logger="udos")
        svc.icm_demo_register(
            {"input_window": [[0.0] * 6] * 6, "result": [0.0] * 6})
        assert any("ICM demo registered" in r.message
                   for r in caplog.records)

    def test_logger_to_stderr_not_stdout(self, svc, capsys):
        """日志只走 stderr, 不串 stdout。"""
        svc.wm_imagine({"window": [[0.0] * 6] * 6, "horizon": 2})
        captured = capsys.readouterr()
        assert "WM latent" not in captured.out, \
            "日志不应出现在 stdout"


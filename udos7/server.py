"""v7 版本化 HTTP 服务（标准库 ThreadingHTTPServer，零额外依赖）。

路由统一前缀 /api/v7：
  GET  /api/v7/health   版本 / 模型加载 / 证据分级
  POST /api/v7/predict  {window, explicit?, horizon?} -> 点预测
  POST /api/v7/interval {window, alpha, explicit?, horizon?} -> 校准区间
  GET  /api/v7/metrics  最近一次 verify 报告摘要（若存在）

区间校准器在首次请求时用独立 calib 轨迹懒拟合并缓存（确定性、CPU 秒级）。
"""
from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, Optional

import torch

from . import __version__
from .contracts import HORIZON, EvidenceGrade
from .dynamics import three_way_splits
from .persistence import load_worldmodel
from .uncertainty import ConformalCalibrator, empirical_coverage

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CKPT = REPO / "checkpoints7" / "worldmodel_v7.0.3.pt"


class V7Service:
    def __init__(self, checkpoint: Optional[Path] = None,
                 n_traj_per_kind: int = 32):
        self.checkpoint = Path(checkpoint) if checkpoint else DEFAULT_CKPT
        self.model = None
        self.meta = {}
        # 校准器缓存键必须含 horizon：不同视界的 q_per_step 形状/数值不同
        self._calib: Dict[tuple, ConformalCalibrator] = {}
        self._splits = None
        self.n_traj = n_traj_per_kind

    def load(self):
        if self.model is None:
            self.model, ckpt = load_worldmodel(self.checkpoint)
            self.meta = {"evidence_grade": ckpt.get("meta", {}).get(
                "evidence_grade", EvidenceGrade.VERIFIED.value)}
        return self

    def _splits_lazy(self):
        if self._splits is None:
            self._splits = three_way_splits(n_traj_per_kind=self.n_traj)
        return self._splits

    def _calibrator(self, use_explicit: bool, horizon: int) -> ConformalCalibrator:
        key = (use_explicit, horizon)
        if key not in self._calib:
            cal = ConformalCalibrator(horizon).fit(
                self.model, self._splits_lazy()["calib"],
                use_explicit=use_explicit)
            self._calib[key] = cal
        return self._calib[key]

    def health(self):
        loaded = self.model is not None
        return {"status": "ok" if loaded or self.checkpoint.exists() else "no_model",
                "version": __version__, "model_loaded": loaded,
                "checkpoint": str(self.checkpoint.name),
                "evidence_grade": EvidenceGrade.VERIFIED.value,
                "api": "v7"}

    @staticmethod
    def _tensors(payload):
        window = torch.tensor(payload["window"], dtype=torch.float32)
        if window.dim() == 2:
            window = window.unsqueeze(0)
        explicit = payload.get("explicit")
        if explicit is not None:
            explicit = torch.tensor(explicit, dtype=torch.float32)
            if explicit.dim() == 1:
                explicit = explicit.unsqueeze(0)
        return window, explicit

    def predict(self, payload: dict) -> dict:
        self.load()
        window, explicit = self._tensors(payload)
        horizon = int(payload.get("horizon", HORIZON))
        traj = self.model.rollout(window, horizon, explicit=explicit)
        return {"version": __version__, "horizon": horizon,
                "trajectory": traj.tolist()}

    def interval(self, payload: dict) -> dict:
        self.load()
        window, explicit = self._tensors(payload)
        horizon = int(payload.get("horizon", HORIZON))
        alpha = float(payload.get("alpha", 0.1))
        use_exp = explicit is not None
        cal = self._calibrator(use_exp, horizon)
        iv = cal.predict_interval(self.model, window, horizon, alpha,
                                  explicit=explicit)
        return {"version": __version__, "alpha": alpha,
                "nominal_coverage": round(1 - alpha, 2),
                "point": iv["point"].tolist(),
                "lower": iv["lower"].tolist(),
                "upper": iv["upper"].tolist()}


def make_handler(service: V7Service) -> type:
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _body(self):
            n = int(self.headers.get("Content-Length", 0))
            return json.loads(self.rfile.read(n) or b"{}")

        def do_GET(self):
            if self.path.rstrip("/") == "/api/v7/health":
                return self._send(200, service.health())
            if self.path.rstrip("/") == "/api/v7/metrics":
                f = REPO / "reports7" / "v7_verification.json"
                if not f.exists():
                    return self._send(404, {"error": "verification report missing"})
                return self._send(200, json.loads(f.read_text()))
            self._send(404, {"error": "unknown route"})

        def do_POST(self):
            try:
                payload = self._body()
                if self.path.rstrip("/") == "/api/v7/predict":
                    return self._send(200, service.predict(payload))
                if self.path.rstrip("/") == "/api/v7/interval":
                    return self._send(200, service.interval(payload))
                self._send(404, {"error": "unknown route"})
            except (ValueError, KeyError) as e:
                self._send(400, {"error": str(e)})
            except FileNotFoundError as e:
                self._send(503, {"error": f"model unavailable: {e}"})

        def log_message(self, *a):
            pass
    return Handler


def serve(host="127.0.0.1", port=8777, checkpoint=None):
    svc = V7Service(checkpoint)
    httpd = ThreadingHTTPServer((host, port), make_handler(svc))
    print(f"UDOS v7 API on http://{host}:{port}/api/v7/ (version {__version__})")
    httpd.serve_forever()


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8777)
    p.add_argument("--checkpoint", default=None)
    a = p.parse_args()
    serve(a.host, a.port, a.checkpoint)

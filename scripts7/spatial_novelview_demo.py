"""v7.3.8 新视角预测演示：多视角空间上下文 vs 单视角（CPU，零第三方依赖）。

产物：reports7/spatial_novelview_demo.json
用法：python scripts7/spatial_novelview_demo.py [训练视角数，默认8]
"""
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from udos7.spatial import run_novelview_benchmark  # noqa: E402
from udos7.observability import Tracer  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "reports7")


def main() -> int:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    tr = Tracer()
    with tr.span("spatial.novelview_benchmark", n_train=n):
        rep = run_novelview_benchmark(n_train=n, tracer=tr)
    print("== UDOS v7.3.8 新视角预测（TSDF 空间上下文，证据 cpu-proto）==")
    print(f"训练视角 {n}，留出新视角 {rep['config']['n_novel_views']}，"
          f"体素 {rep['config']['voxel']}\n")
    print(f"{'上下文':<12}{'轮廓IoU':>10}{'覆盖率':>9}{'精确率':>9}{'深度MAE':>10}")
    for tag, m in rep["results"].items():
        print(f"{tag:<12}{m['silhouette_iou']:>10.3f}{m['coverage']:>9.3f}"
              f"{m['precision']:>9.3f}{m['depth_mae']:>10.3f}")
    print("\n" + rep["finding"])
    ext = rep["external_reference"]
    print(f"\n外部参照（{ext['grade']}）：{ext['source']}")
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "spatial_novelview_demo.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)
    tr.export_jsonl(os.path.join(OUT, "spatial_novelview_demo.traces.jsonl"))
    print(f"\n已写出：{os.path.abspath(path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

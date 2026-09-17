# UDOS v3.5 版本规划 —— SFM 空间基础模型（阶段三·空间）

> 起点：v3.4.5（846 passed / 93% 覆盖率，20 代 checkpoint，ICM 零梯度外挂）
> 终点：v3.5.3
> 训练节点：3.5.0（1 次）
> 迭代总数：恰好 10 节点
> 红线：analogy, not reproduction；CPU-only 合成数据；新能力 opt-in；默认输出逐位等价；846 旧测试只增不删。

## 主线
在 affordance（距离/可达性）和 pce_format（物理Token位置）之上，体系化构建空间基础模型：3D 几何、场景图、占据网格、碰撞检测、坐标变换一致性、空间查询。所有空间推理为零梯度外挂，不改主 predictor 52191 参数。

| # | 版本 | 主题 | 核心交付 | 新增测试 |
|---|------|------|----------|----------|
| 1 | **3.5.0** | SpatialScene 3D 几何核心 + 正式训练 | `udos/spatial.py`：`SpatialObject`（pos3/vel3/radius）、`SpatialScene`（多物体容器）、`SpatialTransform`（平移/旋转/缩放坐标变换，合成可验可逆性）；正式训练 `predictor_v3.5.0.pt`（同口径） | `tests/test_v35_spatial_core.py`：物体/场景结构、变换可逆性、空场景守卫、与 pce_format 接口一致 |
| 2 | 3.5.0.dev1 | SceneGraph 场景图（空间关系） | `udos/scene_graph.py`：`SceneGraph`（物体节点 + 空间关系边：above/below/left/right/near/far/inside，基于几何计算非学习）、关系查询 | `tests/test_v35_scene_graph.py`：关系计算正确、图遍历、空图守卫、与 spatial 接口一致 |
| 3 | 3.5.0.dev2 | OccupancyGrid 占据网格 + 距离场 | `udos/occupancy.py`：`OccupancyGrid`（3D 体素占据，物体半径填充）、`DistanceField`（到最近障碍的有符号距离，CPU 可算） | `tests/test_v35_occupancy.py`：占据填充正确、距离场梯度、空网格守卫、边界条件 |
| 4 | 3.5.0.dev3 | 碰撞检测 + 最近邻 | `udos/collision.py`：`CollisionDetector`（球-球/球-盒代理碰撞，连续帧接触检测）、`NearestNeighbor`（空间最近物体查询，暴力+可选网格加速） | `tests/test_v35_collision.py`：碰撞判定、接触事件、最近邻、空场景守卫、与 spatial 接口一致 |
| 5 | 3.5.0.dev4 | 多视角坐标变换一致性 | `udos/spatial.py` 扩展：多视角投影（正交投影代理，非真实相机）、视角间坐标变换一致性验证（同一物体在不同视角下可互逆映射） | `tests/test_v35_multiview.py`：投影可逆性、跨视角一致性、极端视角守卫 |
| 6 | 3.5.0.dev5 | 空间查询引擎 | `udos/spatial_query.py`：`SpatialQueryEngine`（射线-球相交代理、视线遮挡检测、区域包含查询、范围检索）；复用 occupancy/collision | `tests/test_v35_query.py`：射线相交、视线遮挡、包含查询、空场景守卫 |
| 7 | 3.5.0.dev6 | 空间 A/B + 与 affordance 对比 | A/B：spatial 距离查询 vs affordance 打分的精度/延迟对比；占据网格分辨率 8/16/32 扫描；落 `benchmarks/results/spatial_ab_v3.5.0.json`；收益不稳 opt-in | `tests/test_v35_ab.py`：A/B JSON 落盘、分辨率扫描、opt-in 默认关、被否决候选保留 |
| 8 | **3.5.1** | 集成 + HTTP 端点 | SpatialScene 作为 PhysicalLoopRunner opt-in 空间感知模块；server 新增 `POST /spatial/query`（空间查询）、`POST /spatial/collision`（碰撞检测）；错误语义 400/409/404 | `tests/test_v351_integration.py`：loop+spatial 组合、默认路径逐位一致、两新端点 200/400/409、全特性兼容 |
| 9 | **3.5.2** | 加固 + 21 checkpoint 兼容 + 性能 | backcompat 扩展至 21 件（v2.1.0..v3.5.0）；性能基准 `feature_latency_v3.5.0.json`；全量回归 | `tests/test_v352_service.py`：backcompat 21 件、latency JSON、端点验证、全量回归无退化 |
| 10 | **3.5.3** | Patch 精修 + 文档对齐 | 边界测试（零体积物体、共面碰撞、极端坐标、空查询）；更新 ROADMAP/ARCHITECTURE/DEPLOYMENT/README 至 3.5 线；全量 pytest | `tests/test_v353_edge.py`：边界全绿、全量回归、文档链接有效 |

"""示例物理场景: 一条工厂产线 (机械臂 + 传送带 + 料箱), 含显式因果父节点。"""

from udos.pce_format import PhysicalToken, PhysicsScene


def build_factory_scene(n_steps: int = 12) -> PhysicsScene:
    scene = PhysicsScene(scene_id="factory-line-001", duration=n_steps,
                         metadata={"site": "UDOS-demo-factory"})
    for t in range(n_steps):
        # 传送带匀速前进
        scene.add(PhysicalToken(
            object_id="P-L1-conveyor-002", timestamp=t,
            position=[2.0 + 0.3 * t, 0.0, 0.0],
            velocity=[0.3, 0.0, 0.0],
            attributes={"friction": 0.02, "mass": 12.0},
        ))
        # 机械臂在 t=4 后开始追踪并抓取 (受力变化), 因果父节点为传送带位置
        arm_moving = t >= 4
        scene.add(PhysicalToken(
            object_id="P-L1-robot-arm-001", timestamp=t,
            position=[1.2 + 0.18 * max(t - 4, 0), 0.5, 0.3],
            velocity=[0.18 if arm_moving else 0.0, 0.0, 0.0],
            force=[6.0 if arm_moving else 0.0, 0.0, -9.8],
            attributes={"grip": 0.85 if arm_moving else 0.0, "mass": 4.5},
            causal_parents=([{"object_id": "P-L1-conveyor-002",
                              "timestamp": t - 1}] if arm_moving else []),
        ))
        # 料箱在末端承接
        scene.add(PhysicalToken(
            object_id="P-L2-bin-003", timestamp=t,
            position=[5.5, 0.0, 0.0], velocity=[0.0, 0.0, 0.0],
            attributes={"capacity": 1.0, "fill": min(t / n_steps, 1.0)}))
    return scene


def build_scene_chunks(n_chunks: int = 3, step_per_chunk: int = 4):
    """把长场景切成多个分块, 用于 GPM chunking 聚合演示。"""
    full = build_factory_scene(n_steps=n_chunks * step_per_chunk)
    chunks = []
    for i in range(n_chunks):
        lo, hi = i * step_per_chunk, (i + 1) * step_per_chunk
        chunk = PhysicsScene(scene_id=f"{full.scene_id}#chunk{i}")
        chunk.tokens = [t for t in full.tokens if lo <= t.timestamp < hi]
        chunks.append(chunk)
    return full, chunks

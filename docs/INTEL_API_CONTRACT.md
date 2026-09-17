# UDOS v5.0.2 情报分析服务 — API 契约（锁定）

> 底层智能内核驱动 AGI/ASI 倒计时网站。纯逻辑/零梯度/零训练/CPU 离线可跑。
> 公式口径与 `agi-asi-countdown/backend/app/core` 对齐（只 Read 不改网站项目）。

## 启动
```bash
UDOS_AUTH=off python -m udos.server --port 8000 --preset small
# 默认离线确定性 seed，无需 LLM key / GPU
```

## 端点（9 个，字段名锁定）

### GET /intel/health
```json
{"service": "udos-intel", "engine_version": "5.0.2", "mode": "offline-deterministic"}
```

### GET /intel/coefficient?days=N
```json
{"series": [{"date", "raw_score", "weighted_score", "formula_version",
             "milestone_flag", "confidence", "sources": []}]}
```

### POST /intel/coefficient/step
请求：`{"events": [{"kind","impact","credibility","independent_sources","category","date","sources"}], "human_confirmed": bool}`
响应：`{"weighted_score", "formula_version", "gate": {"triggered", "reasons": [], "human_required": true}}`
- 奇点门：impact≥9 且 category∈{core_tech,paradigm_shift} 且 independent_sources≥2 且 human_confirmed 才触发；单点/单源/未确认一律不触发。

### GET /intel/capability
```json
{"dimensions": [{"name","score","sources":[],"confidence"}]}
```

### GET /intel/routes
```json
{"routes": [{"route","orgs":[],"people":[],"progress_pct",
             "eta_median","eta_earliest","eta_latest","confidence","milestones":[]}]}
```
- `eta_earliest`=最乐观最早；`eta_latest`=最悲观最晚；保证 earliest≤median≤latest。

### GET /intel/predictions?status=pending|verified|falsified
```json
{"predictions": [{"author","org","claim","predicted_on","due_date","source_url","source_type","status","score","related_route"}]}
```

### POST /intel/predictions/score
请求：`{"predictions": [{"id"?: str, "probability": float∈[0,1], "outcome": int∈{0,1}|缺失}]}`
响应：`{"scored": [{"id","probability","outcome","score","status"}]}`
- score=(probability-outcome)²（Brier，0 完美 1 最差）；outcome=1→verified，0→falsified；
  outcome 缺失→status=pending 且 score=null；probability 越界/空 predictions→400。

### GET /intel/models
```json
{"models": [{"model_id","label","filter_rule","perspective(leader|researcher|community)",
             "bias_adjustment","median_eta","series":[{"date","eta_year"}]}]}
```
- 多模型聚合：领袖视角施加乐观偏差折减（eta 后移 +1.5 年），研究者并列。

### GET /intel/asi
```json
{"rsi_indicators":[{"name","value"}], "takeoff_scenarios":[...],
 "asi_eta":{"median","earliest","latest","confidence","disclaimer":"假设非事实"},
 "risk_signals":[{"date","signal","capability_or_risk","source_url"}]}
```

## 错误语义
- 缺字段/非法 → 400；未知路由 → 404；未挂载/无数据 → 409；外部依赖缺失 → 503。
- `UDOS_AUTH=on`：未认证 401、越权 403（viewer 可读 GET、operator 可写 POST）；off 全透明。

## 离线 seed
`udos/intelligence/seed.py` 内置确定性演示数据，对齐网站 snapshots 12 个 JSON 结构，无 GPU/API key 即可运行。

### GET /intel/infrastructure（始终 200，不受 KVCACHE 开关影响）
```json
{"kvcache_enabled","gpu_available":false,"qat":"ENV_BLOCKED",
 "hit_rate","recompute_units","ttft_proxy_ms","mode",
 "disclaimer":"CPU 机制原型、非真实 GPU/QAT 数据、非厂商数字"}
```

### GET /intel/brain（始终 200）
```json
{"enabled":true,"gpu_available":false,"neural_simulator":"ENV_BLOCKED",
 "disclaimer":"CPU 机制原型, 非真实生物/GPU 数据"}
```

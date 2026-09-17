"""把内嵌 catalog 装配成 ResourceRegistry (全部 79 条)。

诚实分级:
  L3 requires_weights: 大 VLA / 世界模型 / 真实权重 —— 契约就绪, 本环境不下载。
  L2 requires_pkg:      CPU 可装小库, 惰性真实 import (yourdfpy 已装并 smoke)。
  L1:                   纯 python 格式适配 + 通用轨迹归一化 (inline fixture 往返)。
"""
from __future__ import annotations

from typing import Dict

from ..resource_registry import ResourceConnector, ResourceRegistry
from ._catalog_data import CATALOG_ROWS
from .base import GenericConnector, build_spec
from . import specialized

# ---- L3: 大权重/GPU 模型 (本环境不下载, 仅契约) ----
L3_WEIGHTS = {
    "openvla", "octo_octo_small_octo_base", "openpi_0_0_fast_0_5",
    "rdt_1b_robotics_diffusion_transformer", "cogact", "smolvla",
    "tinyvla", "mobilevla", "gr_1_gr_2_gr00t", "v_jepa_v_jepa_2_meta",
    "nvidia_cosmos", "genie_deepmind", "physbrain_1_5_deepcybo",
    "unifolm_wla_1_0_unitree", "wall_wm_wall_oss_wall_ss_x_square",
    "ctm_ctm_imagenet_sakana_ai", "doc_to_lora_d2l", "skild_s1",
    "generalist_gen_1_5", "mimicdroid_dreamdojo_icm",
}

# ---- L2: CPU 可装小库 (惰性 import) ----
L2_PKG: Dict[str, str] = {
    "yourdfpy": "yourdfpy",
    "pytorch_kinematics": "pytorch_kinematics",
}

# ---- 专门 L1 格式适配器映射 ----
ACTION_CHUNK = {"act_action_chunking_with_transformers_aloha", "diffusion_policy"}
EPISODE_SCHEMA = {
    "lerobot_framework_incl_smolvla_so_100", "lerobot_hf_lerobot",
    "open_x_embodiment_rt_x", "rt_x_open_x_embodiment_rt_1_x_rt_2_x",
    "robomimic_mimicgen", "robomimic", "droid", "bridgedata_v2",
    "libero", "calvin", "aloha_mobile_aloha", "rh20t", "robomind_robomind_2_0",
    "agibot_world", "unitree_open_datasets_bitrobot_hiw_500",
}
SMPL_POSE = {"amass_smpl_smpl_x", "grab_grasping_actions_with_bodies",
             "humanplus", "omnih2o", "exbody", "exbody_2"}
BVH = {"lafan1", "unitree_lafan1_retargeting_dataset_human_as_humanoid_primeu"}
VLA_DISCRETE = {"openvla"}
RSSM = {"dreamer_dreamerv3"}


def _connector_for(row: dict) -> ResourceConnector:
    rid = row["id"]
    if rid in L2_PKG:
        spec = build_spec(row, level="L2", requires_pkg=L2_PKG[rid])
        return specialized.URDFFKConnector(spec)
    if rid in L3_WEIGHTS:
        spec = build_spec(row, level="L3",
                          requires_gpu=True, requires_weights=True)
        if rid in VLA_DISCRETE:
            return specialized.VLADiscreteConnector(spec)
        return GenericConnector(spec)
    # L1 专门格式
    if rid in ACTION_CHUNK:
        return specialized.ActionChunkingConnector(build_spec(row, level="L1"))
    if rid in EPISODE_SCHEMA:
        return specialized.EpisodeSchemaConnector(build_spec(row, level="L1"))
    if rid in SMPL_POSE:
        return specialized.SMPLPoseConnector(build_spec(row, level="L1"))
    if rid in BVH:
        return specialized.BVHClipConnector(build_spec(row, level="L1"))
    if rid in VLA_DISCRETE:
        return specialized.VLADiscreteConnector(build_spec(row, level="L1"))
    if rid in RSSM:
        return specialized.WorldModelRSSMConnector(build_spec(row, level="L1"))
    # 默认 L0/L1: 元数据 + 通用轨迹归一化
    return GenericConnector(build_spec(row, level="L1"))


def build_default_registry(profile: str = "performance") -> ResourceRegistry:
    reg = ResourceRegistry(profile=profile)
    for row in CATALOG_ROWS:
        reg.register(_connector_for(row))
    return reg

model_config_pretune = {
    "use_cluster": True,
    "freeze": False,
    "vision_tune": False,

    "spatial_cluster_rate0": 64,  # 0.25
    "spatial_cluster_rate1": 32,  # 0.5
    "spatial_cluster_rate2": 16,  # 0.5

    "temporal_cluster_rate": 1/16,
}

model_config_finetune = {
    "use_cluster": True,
    "freeze": False,
    "mm_tune": True,
    "vision_tune": False,

    "spatial_cluster_rate0": 64,  # 0.25
    "spatial_cluster_rate1": 32,  # 0.5
    "spatial_cluster_rate2": 16,  # 0.5

    "temporal_cluster_rate": 1/16,
}


## don't do token merging

model_config_v0 = {
    "cluster_type": "v0",
    "use_cluster": False,
}


## wpq: below are model config.

# original impl
model_config_v1 = {
    "cluster_type": "v1",
    "use_cluster": True,
    "spatial_cluster_rate0": 64,  # 0.25
    "spatial_cluster_rate1": 32,  # 0.5
    "spatial_cluster_rate2": 16,  # 0.5
    "temporal_cluster_rate": 1/16,
    "coord_weight": 0,
}

# reimpl: temporal->spatial->video token merging.
model_config_v2 = {
    "cluster_type": "v2",
    "use_cluster": True,
    "sample_ratios_temporal": [1/16],
    "sample_ratios_spatial": [64, 32, 16],
    "sample_ratios_video": [64, 32, 16],
    "coord_weights": [0., 0., 0.],
    "token_orderings": ['raster', 'default', 'default'],
    "prune_ratios_spatial": None,
    "prune_ratios_video": None,
    "flow": "sequential",
    "pe_type": None,
}


# 1 single 3d token merging.
model_config_v3 = {
    "cluster_type": "v3",
    "use_cluster": True,
    "sample_ratios_spatial": [64, 32, 16],
    "sample_ratios_video": [64, 32, 16],
    "coord_weights": [0, 0],
    "token_orderings": ["default", "default"],
    "prune_ratios_spatial": None,
    "prune_ratios_video": None,
    "flow": "sequential",
    "pe_type": None,
}


model_config_v4 = {
    "cluster_type": "v4",
    "use_cluster": True,
    "matryoshka_vis_token_scale": None,
}
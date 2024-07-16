from abc import ABC, abstractmethod
import math
import numpy as np
import torch
import torch.nn as nn
from .multimodal_encoder.builder import build_vision_tower
from ChatUniVi.constants import *
from .cluster import CTM, TCBlock, TokenMergeClusterDPCKNN, VideoTokenMergeClusterDPCKNN, create_token_dict_from_features
from collections import OrderedDict
import collections
from .multimodal_projector.builder import build_vision_projector


class MetaModel:
    def __init__(self, config):
        super(MetaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            self.vision_tower = build_vision_tower(config, delay_load=True)
            self.mm_projector = nn.Linear(config.mm_hidden_size, config.hidden_size)

        if hasattr(config, "config"):
            self.initialize_cluster_modules(config.config)
        else:
            self.use_cluster = False

    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter

        self.config.mm_vision_tower = vision_tower

        vision_tower = build_vision_tower(model_args)

        self.config.use_mm_proj = True
        self.config.mm_hidden_size = vision_tower.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature

        if fsdp is not None and len(fsdp) > 0:
            self.vision_tower = [vision_tower]
        else:
            self.vision_tower = vision_tower

        if not hasattr(self, 'mm_projector'):
            self.mm_projector = build_vision_projector(self.config)

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))


    def initialize_cluster_modules(self, model_config):
        """
            call the corresponding init function, 
                e.g., if `model_config.cluster_type` is "v1", then calls 
                    `self.initialize_cluster_modules_v1()`

            assumes `model_config` contains ["use_cluster", "cluster_type"]
        """

        self.use_cluster = model_config["use_cluster"]
        if not self.use_cluster:
            return
        self.cluster_type = model_config.get('cluster_type', 'v1')

        method_name = f"initialize_cluster_modules_{self.cluster_type}"
        if getattr(self, method_name, None) is None:
            print(f"[MetaModel.initialize_cluster_modules] {method_name} not implemented.")
        else:
            getattr(self, method_name)(model_config)


    def initialize_cluster_modules_v1(self, model_config):
        """handles both default model (chatunivi hf ckpt) & modified code with coord_weight. """

        coord_weight = model_config.get("coord_weight", 0)

        self.ctm0 = CTM(sample_ratio=model_config["spatial_cluster_rate0"], embed_dim=self.config.mm_hidden_size, dim_out=self.config.mm_hidden_size, k=5, coord_weight=coord_weight)
        self.block0 = TCBlock(dim=self.config.mm_hidden_size, num_heads=8)

        self.ctm1 = CTM(sample_ratio=model_config["spatial_cluster_rate1"], embed_dim=self.config.mm_hidden_size, dim_out=self.config.mm_hidden_size, k=3, coord_weight=coord_weight)
        self.block1 = TCBlock(dim=self.config.mm_hidden_size, num_heads=8)

        self.ctm2 = CTM(sample_ratio=model_config["spatial_cluster_rate2"], embed_dim=self.config.mm_hidden_size, dim_out=self.config.mm_hidden_size, k=3, coord_weight=coord_weight)
        self.block2 = TCBlock(dim=self.config.mm_hidden_size, num_heads=8)

        self.ctm3 = CTM(sample_ratio=model_config["temporal_cluster_rate"], embed_dim=self.config.mm_hidden_size, dim_out=self.config.mm_hidden_size, k=5, coord_weight=coord_weight)
        self.block3 = TCBlock(dim=self.config.mm_hidden_size, num_heads=8)

    def initialize_cluster_modules_v2(self, model_config):
        """Replicate original implementation myself. """
        ks = [5, 3, 3]
        self.token_merging_model = VideoTokenMergeClusterDPCKNN(
            sample_ratios_temporal=model_config['sample_ratios_temporal'],
            sample_ratios_spatial=model_config['sample_ratios_spatial'],
            sample_ratios_video=model_config['sample_ratios_video'],
            ks=ks,
            coord_weights=model_config['coord_weights'],
            token_orderings=model_config['token_orderings'],
            prune_ratios_spatial=model_config.get('prune_ratios_spatial', None),
            prune_ratios_video=model_config.get('prune_ratios_video', None),
            flow=model_config.get('flow', 'sequential'),
        )

    def initialize_cluster_modules_v3(self, model_config):
        """ 1 single 3d token merging module """
        coord_weight_spatial, coord_weight_video = model_config["coord_weights"]
        token_ordering_spatial, token_ordering_video = model_config["token_orderings"]
        sample_ratios_spatial = model_config['sample_ratios_spatial']
        sample_ratios_video = model_config['sample_ratios_video']
        ks = [5,3,3]
        self.token_merge_image = TokenMergeClusterDPCKNN(
            sample_ratios=sample_ratios_spatial,
            ks=ks[:len(sample_ratios_spatial)],
            coord_weight=coord_weight_spatial,
            token_ordering=token_ordering_spatial,
            prune_ratios=model_config.get('prune_ratios_spatial', None),
            flow=model_config.get('flow', 'sequential'),
        )
        self.token_merging_video = TokenMergeClusterDPCKNN(
            sample_ratios=sample_ratios_video,
            ks=ks[:len(sample_ratios_video)],
            coord_weight=coord_weight_video,
            token_ordering=token_ordering_video,
            prune_ratios=model_config.get('prune_ratios_video', None),
            flow=model_config.get('flow', 'sequential'),
        )


def position_encoding_multidim(P, D, div_base=10_000., ordering='interleave'):
    """Given a set of points `P`, generate the corresponding position embedding of length `D`
        Note each dimension in `P` uses `D/ndim` number of channels.
            e.g., If ndim=3, D=4096, then use 4096/3~1365 channels to embed x coordinate.
        `P`     (N1, ..., Nd, ndim)
        Returns (N1, ..., Nd, D)
    """
    import numpy as np
    prefix_dims = list(P.shape[:-1])
    ndim = P.shape[-1]
    device = P.device
    step = 2*ndim
    div_term = torch.exp(torch.arange(0, D, step, device=device, dtype=P.dtype) * -(math.log(div_base) / D))
    L = []
    for di in range(ndim):
        x = P[...,[di]] * div_term
        L += [ torch.sin(x), torch.cos(x) ]
    if ordering == 'interleave':
        E = torch.stack(L, dim=P.ndim)
    elif ordering == 'concat':
        E = torch.cat(L, dim=P.ndim-1)
    # if D%ndim!=0, then need to truncate end.
    E = E.reshape(*prefix_dims, -1)[...,:D]
    return E


def ravel_multi_index(coords, shape):
    r"""Converts a tensor of coordinate vectors into a tensor of flat indices.

    This is a `torch` implementation of `numpy.ravel_multi_index`.

    Reference: https://github.com/pytorch/pytorch/issues/35674

    Args:
        coords: A tensor of coordinate vectors, (*, D).
        shape: The source shape.

    Returns:
        The raveled indices, (*,).
    """

    shape = coords.new_tensor((*shape, 1))
    coefs = shape[1:].flipud().cumprod(dim=0).flipud()

    return (coords * coefs).sum(dim=-1)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    # x: (B, #heads, L, D)
    # for each (b, head, l), create two halfs q = (q0, q1, q2, q3) and returns (-q2, -q3, q0, q1)
    # as complex number: (q0+i*q2, q1+i*q3)
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def rope3d(X, P, num_heads=1, theta_type='1d', divbase=10_000., channel_partition='interleave'):
    """Apply multi-dim positions `P` to features `X` via RoPE.
        This is somewhat in line with how huggingface's llama impl that 
            - treat (x0, xd/2-1) as a complex number.
            - applies rope to dimension of embedding of each head.

        `X`    (B, L, D_hidden)
            either key or query vector
        `P`    (B, L, ndim) 
            - if ndim=1 and theta_type='1d', reduces to 1D RoPE.
            - if ndim>1, then encodes multi-dimensional positions into features `X`
    """
    device = X.device
    dtype = X.dtype
    B, L, D_hidden = X.shape
    ndim = P.shape[-1]
    # (B, L, D_hidden) -> (B, L, #heads, D) -> (B, #heads, L, D) where D is dimension of each head.
    X = X.view(B, L, num_heads, D_hidden//num_heads).transpose(1, 2)
    D = X.shape[-1]

    # inv_freq: (D//2,)
    # 1d: θ_0, θ_1, ... θ_d/2-1
    # tied: θ_z0, θ_y0, θ_x0, θ_z1, θ_y1, θ_x1, ..., θ_zd/6, θ_yd/6, θ_xd/6  (channel_partition == 'interleave')
    # tied: θ_z0, θ_z1, ..., θ_zd/6, ..., θ_x0, θ_x1, ..., θ_zx/6  (channel_partition == 'sequential')
    if theta_type == '1d':
        inv_freq = 1.0 / (divbase ** (torch.arange(0, D, 2).float().to(device) / D))
    elif theta_type == 'tie':
        inv_freq = 1.0 / (divbase ** (torch.arange(0, D, 2*ndim).float().to(device) / D))
        inv_freq = inv_freq.repeat_interleave(ndim) if channel_partition == 'interleave' else inv_freq.repeat(ndim)
        inv_freq = inv_freq[:(D // 2)]
    else:
        raise ValueError(f'Invalid theta_type={theta_type}')
    # (B, L, D//2): freqs = torch.einsum("...i,j->...ij", P, inv_freq) # outer product
    rep = (D//2) // ndim
    rem = (D//2)  % ndim
    # (B, L, 3) -> (B, L, D//2) where last dim repeat [z,y,x] -> [z,y,x,z,y,x,...,z,y,x]
    P_expand = torch.cat((
        P.repeat((1,)*len(P.shape[:-1]) + (rep,)) if channel_partition == 'interleave' else P.repeat_interleave(rep, dim=-1),
        torch.zeros(P.shape[:-1] + (rem,), dtype=dtype, device=device)), dim=-1)
    # (B, L, D//2) * (D//2,) -> (B, L, D//2)
    freqs = (P_expand * inv_freq)
    # (B, L, D)
    emb = torch.cat((freqs, freqs), dim=-1) # each embedding: i, i+D//2 is same 
    cos = emb.cos()[:, None, ...].to(dtype=dtype)
    sin = emb.sin()[:, None, ...].to(dtype=dtype)
    # for each batch item, each head in MHA, do the following operation on (L, D)
    # (B, #heads, L, D)
    X = (X * cos) + (rotate_half(X) * sin)
    # (B, #heads, L, D) -> (B, L, D_hidden)
    X = X.transpose(1, 2).reshape(B, L, D_hidden)

    return X


def regularize_covariance(cov, epsilon=1e-6):
    regularization = epsilon * torch.eye(cov.shape[-1], device=cov.device)
    return cov + regularization

def sample_truncated_normal(mean, covariance, vol_shape=None, num_samples=1, truncation_stddev_factor=2, epsilon=1e-6):
    """Sample from 3D Gaussian defined by `mean` and `covariance`.
            `mean`    (N1, ..., Nd, 3)
            `cov`     (N1, ..., Nd, 3, 3)
            Returns   (#samples, N1, ..., Nd, 3)
                samples from 3d gaussian
    """
    import torch.distributions as dist
    device = mean.device
    if vol_shape is not None:
        vol_shape = torch.tensor(vol_shape, device=device)
    shape_start = mean.shape[:-1]
    mean = mean.reshape(-1, 3)
    covariance = covariance.reshape(-1, 3, 3)
    all_samples = torch.zeros((num_samples, mean.shape[0]) + (3,), device=device)
    for idx in range(mean.shape[0]):
        current_mean = mean[idx]
        current_cov = covariance[idx]
        for i in range(math.ceil(math.log(1/epsilon, 10)) + 1):
            try:
                mvn = dist.MultivariateNormal(current_mean, current_cov)
            except:
                current_cov = regularize_covariance(current_cov, epsilon=epsilon*(10**i))
        samples = []
        while len(samples) < num_samples:
            sample = mvn.sample()  # (3,)
            std_devs = (sample - current_mean).abs() / torch.sqrt(torch.diag(current_cov))  # (3,)
            if torch.all(std_devs <= truncation_stddev_factor):
                if vol_shape is not None and torch.all(sample >= 0) and torch.all(sample < vol_shape):
                    samples.append(sample)
                elif vol_shape is None:
                    samples.append(sample)
        samples = torch.stack(samples)  # (num_samples, 3)
        all_samples[:, idx, :] = samples
    all_samples = all_samples.reshape((num_samples,) + shape_start + (3,))
    return all_samples


def treat_z_as_event_id(token_merging_outputs, T, config):
    # has to return events. should not be used for module that does not group frames to events/segments.
    # - note this also modifies `token_merging_outputs['cluster_means/covs']` in place.
    # - note should also set kvs['vidmaxpos'] properly.
    if not ('events' in token_merging_outputs and 
            'sample_ratios_temporal' in config.config 
            and 'cluster_means' in token_merging_outputs 
            and 'cluster_covs' in token_merging_outputs):
        return token_merging_outputs, T
        
    events = token_merging_outputs['events']
    sample_ratio_temporal = config.config['sample_ratios_temporal'][0]
    cluster_means = token_merging_outputs['cluster_means']
    cluster_covs = token_merging_outputs['cluster_covs']
    ## update time dimension size to be `num_events`
    T = max(math.ceil(T * sample_ratio_temporal), 1)
    ## adjust mean/cov properly after scale s.t. z is event_id instead of frame number
    # (3, 3): scale z (first) direction by sample_ratio_temporal
    A = torch.diag(torch.tensor([sample_ratio_temporal if sample_ratio_temporal <= 1 else 1/sample_ratio_temporal, 1, 1], device=cluster_means.device, dtype=cluster_means.dtype))
    # apply to mean. Aμ & covariance AΣAᵀ. cluster_means = cluster_means @ A.T
    cluster_covs = (A @ cluster_covs @ A.T)
    ## since event may consistute non-consecutive frames, force z to be [0, ..., num_events-1]
    # (B, N) convert z-axis value to the corresponding event_id
    cluster_means_z = torch.floor(cluster_means[...,0]).to(torch.long)
    for event_id, event_frame_ids in enumerate(events):
        cluster_means_z[torch.isin(cluster_means_z, torch.tensor(event_frame_ids, device=cluster_means_z.device))] = event_id
    cluster_means[...,:, 0] = cluster_means_z + 1/2 # consistent with convention.
    token_merging_outputs['cluster_means'] = cluster_means 
    token_merging_outputs['cluster_covs'] = cluster_covs
    
    return token_merging_outputs, T



class ChatUniViMetaForCausalLM(ABC):
    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def encode_images(self, images):
        image_features = self.get_model().get_vision_tower()(images, select_feature="patch")
        return image_features

    # def positional_encoding(self, x, num_features=1024, max_len=64):
    #     p = torch.zeros((1, max_len, num_features))
    #     _x = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1) / torch.pow(10000,
    #                                                                         torch.arange(0, num_features, 2, dtype=torch.float32) / num_features)

    #     p[:, :, 0::2] = torch.sin(_x)
    #     p[:, :, 1::2] = torch.cos(_x)
    #     x = x + p[:, :x.shape[1], :].to(x.device).to(x.dtype)
    #     return x

    def positional_encoding(self, image_token_embeds, token_merging_outputs, input_type):
        """
            image_token_embeds
                image: (B, #tokens, 4096)
                    B=1 for video & arbitrary for image. this function works for both.
        """
        from rosemary import parse_kv_from_string

        B, N, _ = image_token_embeds.shape
        device = image_token_embeds.device

        outputs = {
            "image_token_embeds": image_token_embeds,
            "position_ids": torch.arange(N, device=device, dtype=torch.long).reshape(1, N).repeat(B, 1),
            "max_positions": torch.tensor(N, device=device),
        }

        ## only add position_encoding if use token clustering & pe_type is supplied.
        config = self.get_model().config
        if not hasattr(config, "config") or config.config.get('pe_type', None) is None:
            return outputs

        pe_type = config.config['pe_type']
        kvs = parse_kv_from_string(pe_type)

        if kvs[0] == 'apesinu':
            if kvs['enc'] == 'mean':
                if 'cluster_means' not in token_merging_outputs: return outputs
                # (B, N, 3) where e.g. N=64+32=16
                P = token_merging_outputs['cluster_means']
            elif kvs['enc'] == 'mean+diagcov':
                if 'cluster_means' not in token_merging_outputs or 'cluster_covs' not in token_merging_outputs: return outputs
                cluster_covs_diag = torch.stack([
                    token_merging_outputs['cluster_covs'][:, :, 0, 0],
                    token_merging_outputs['cluster_covs'][:, :, 1, 1],
                    token_merging_outputs['cluster_covs'][:, :, 2, 2],
                ], dim=-1)
                # (B, N, 6)
                P = torch.cat((
                    token_merging_outputs['cluster_means'],
                    cluster_covs_diag,
                ), dim=2)
            elif kvs['enc'] == 'pos1d':
                P = torch.arange(N, device=device, dtype=image_token_embeds.dtype).reshape(1, -1, 1).repeat(B, 1, 1)
            else:
                raise ValueError(f"[positional_encoding] enc={kvs['enc']} not implemented.")

            ## whether to discretize the position or not.
            discrete_positions = bool(kvs.get('discrete', 0))
            if discrete_positions is True:
                P = torch.floor(P)

            ## re-order tokens before adding position encoding.
            token_ordering = kvs.get('tokord', None)
            if token_ordering is not None:
                ndim = P.shape[-1]
                # sort_score: (B, N)
                if token_ordering == "random":
                    sort_score = torch.rand_like(P[...,0])
                elif token_ordering == "raster":
                    # maps multi-dimensional coordinate [z,y,x] -> z*(64*24)+10*y+x for sorting.
                    # (3,)
                    P_max = torch.stack([torch.max(P[...,i]) + 1 for i in range(ndim)]).squeeze()
                    # -> (16*16, 16, 1) 
                    P_max_cumprod_rev = torch.cat((
                        torch.cumprod(P_max[1:].flip(dims=[0]), dim=0).flip(dims=[0]),
                        torch.tensor([1.], device=P_max.device, dtype=P_max.dtype)
                    ))
                    # (B, N)
                    sort_score = torch.stack(
                        [P_max_cumprod_rev[di] * P[..., di] for di in range(ndim)], dim=-1
                    ).sum(dim=-1)
                else:
                    raise ValueError(f"[positional_encoding] token_ordering={token_ordering} not implemented.")
                # [new_cluster_id: old_cluster_id]
                inds = torch.argsort(sort_score, dim=-1, descending=False)
                # (B, N, 3): re-ordered `P`
                P = torch.stack([
                    torch.take_along_dim(P[b], inds[b, :, None].repeat(1, ndim), dim=0) for b in range(B)
                ]).reshape_as(P)
                # (B, \#, D)
                image_token_embeds = torch.stack([
                    torch.take_along_dim(image_token_embeds[b], inds[b, :, None].repeat(1, image_token_embeds.shape[-1]), dim=0) for b in range(B)
                ]).reshape_as(image_token_embeds)
            
            ## add position encoding
            ordering = kvs.get('ord', 'interleave')
            div_base = float(kvs.get('divbase', 10_000.))
            # image: (B, N, 4096)
            pe = position_encoding_multidim(P, image_token_embeds.shape[-1], div_base=div_base, ordering=ordering)
            pe = pe.to(image_token_embeds.device).to(image_token_embeds.dtype)
            image_token_embeds = image_token_embeds + pe
            
            outputs.update({
                "image_token_embeds": image_token_embeds,
            })
        elif kvs[0] == 'rope3d':
            T = MAX_IMAGE_LENGTH

            event_id_as_z = bool(kvs.get('eventidasz', 0))
            if event_id_as_z and input_type == 'video':
                token_merging_outputs, T = treat_z_as_event_id(token_merging_outputs, T, config)

            image_token_embeds = rope3d(
                image_token_embeds,
                token_merging_outputs['cluster_means'] - 1/2,
                num_heads=kvs.get('nh', 32),
                theta_type=kvs.get('thetatype', '1d'),
                divbase=kvs.get('divbase', 10_000.),
                channel_partition=kvs.get('chpartition', 'interleave'),
            )
            outputs.update({
                "image_token_embeds": image_token_embeds,
            })
        elif kvs[0] == 'rope1d':
            
            T = MAX_IMAGE_LENGTH
            H = W = int(self.get_model().get_vision_tower().config.image_size / self.get_model().get_vision_tower().config.patch_size)

            event_id_as_z = bool(kvs.get('eventidasz', 0))
            if event_id_as_z and input_type == 'video':
                token_merging_outputs, T = treat_z_as_event_id(token_merging_outputs, T, config)

            if kvs['enc'] == 'pos3dravel':
                if 'cluster_means' not in token_merging_outputs: return outputs
                # (B, N, 3)
                P = token_merging_outputs['cluster_means']
                P = torch.floor(P).to(torch.long) # discretize
                # (B, N)
                P = ravel_multi_index(P, (1, H, W)) # note T length does not matter when ravel multi-index
            elif kvs['enc'] == 'pos3dsampleravel':
                if 'cluster_means' not in token_merging_outputs or 'cluster_covs' not in token_merging_outputs: return outputs
                truncation_stddev_factor = kvs.get('truncstd', 2)
                # (B, N, 3)
                P = sample_truncated_normal(
                    token_merging_outputs['cluster_means'],
                    token_merging_outputs['cluster_covs'],
                    vol_shape=(T, H, W),
                    truncation_stddev_factor=truncation_stddev_factor,
                    num_samples=1)[0]
                # (B, N)
                P = ravel_multi_index(P, (1, H, W)) # note T length does not matter when ravel multi-index
            elif kvs['enc'].startswith('pos3dhilbert'):
                if 'cluster_means' not in token_merging_outputs: return outputs
                # (B, N, 3)
                P = token_merging_outputs['cluster_means']
                if getattr(self, 'hilbert_curve', None) is None:
                    from metasummer2024.gilbert3d import gilbert3d
                    # (T*H*W, 3)
                    if kvs['enc'] == 'pos3dhilbert2':
                        # the same 2d hilbert curve, for each fram.
                        self.hilbert_curve = torch.tensor(list(gilbert3d(1, H, W)), dtype=P.dtype, device=P.device) + 1/2
                        self.hilbert_curve = torch.cat([self.hilbert_curve + torch.tensor([[t, 0, 0]], device=P.device, dtype=P.dtype) for t in range(T)], dim=0)
                    else:
                        self.hilbert_curve = torch.tensor(list(gilbert3d(T, H, W)), dtype=P.dtype, device=P.device) + 1/2
                # (B, N, T*H*W)
                distances = torch.cdist(P, self.hilbert_curve[None, ...].repeat(P.shape[0], 1, 1))
                # (B, N)
                P = distances.argmin(dim=-1)
            elif kvs['enc'] == 'pos1d':
                # (B, N)
                P = torch.arange(N, device=device, dtype=torch.float32).reshape(1, N).repeat(B, 1)
            else:
                raise ValueError(f"[positional_encoding] enc={kvs['enc']} not implemented.")

            # max(P) can be at most 64*24*24=36,864 tokens for video. Normalize to kvs['vidmaxpos']
            scale_position_ids = bool(kvs.get('scalepos', 0))
            if scale_position_ids:
                if input_type == 'image':
                    P = P / (H*W / kvs['imgmaxpos'])
                elif input_type == 'video':
                    P = P / (T*H*W / kvs['vidmaxpos'])
                else:
                    raise ValueError(f'[positional_encoding] invalid input_type={input_type}')
            # (B, N)
            position_ids = torch.round(P).to(torch.long)

            ## re-order tokens by position
            token_ordering = kvs.get('tokord', None)
            if token_ordering is not None and token_ordering != 'default':
                # sort_score: (B, N)
                if token_ordering == "random":
                    sort_score = torch.rand_like(P)
                elif token_ordering in ("raster", "sort"):
                    sort_score = P
                else:
                    raise ValueError(f"[positional_encoding] token_ordering={token_ordering} not implemented.")
                inds = torch.argsort(sort_score, dim=-1, descending=False)
                # (B, N)
                position_ids = torch.take_along_dim(position_ids, inds, dim=-1)
                # (B, N, D)
                image_token_embeds = torch.stack([
                    torch.take_along_dim(image_token_embeds[b], inds[b, :, None].repeat(1, image_token_embeds.shape[-1]), dim=0) for b in range(B)])

            # decides what next position_id should be after image tokens
            max_positions_type = kvs.get('maxpostype', 'add1')
            if  max_positions_type is not None:
                if max_positions_type == 'add1': 
                    # if position_ids=[0,1,2], then position_ids_max=3
                    # corresponds to default case, if position_ids consecutive, then it's simply #tokens
                    max_positions = position_ids.max()+1
                elif max_positions_type == 'max':
                    # allocate `kvs['videomaxpos']` position_ids to represent video
                    max_positions = torch.tensor(kvs['vidmaxpos'] if input_type == 'video' else kvs['imgmaxpos'], device=device)
                else:
                    raise ValueError(f'[positional_encoding] invalid max_positions_type={max_positions_type}')

            outputs.update({
                "image_token_embeds": image_token_embeds,
                "position_ids": position_ids,
                "max_positions": max_positions,
            })
        else:
            raise ValueError(f"[positional_encoding] pe_type={pe_type} not implemented.")
        
        return outputs


    def project(self, image_features, input_type="image", matryoshka_vis_token_scale=None):
        """
            call the corresponding `project` function, 
                e.g., if `self.get_model().cluster_type` is "v1", then calls 
                    `self.project_v1()`
        """
        if not self.get_model().use_cluster:
            if input_type == "video":
                image_features, cls_features = torch.mean(image_features, dim=0, keepdim=False).unsqueeze(
                    0), torch.mean(image_features, dim=1, keepdim=False).unsqueeze(0)
                image_features = torch.cat([image_features, cls_features], dim=1)
            image_token_embeds = self.get_model().mm_projector(image_features)
            token_merging_outputs = None
        else:
            cluster_type = self.get_model().cluster_type
            method_name = f"project_{cluster_type}"
            if not getattr(self, method_name):
                raise ValueError(f"[ChatUniViMetaForCausalLM.project] {method_name} not supported.")
            if (cluster_type == 'v4' and not matryoshka_vis_token_scale) or (cluster_type != 'v4' and matryoshka_vis_token_scale):
                raise ValueError('Only use `matryoshka_vis_token_scale` when `cluster_type="v4"`.')
            
            if cluster_type == "v1":
                image_features = getattr(self, method_name)(image_features, input_type=input_type)
                token_merging_outputs = None
            elif cluster_type == "v4":
                image_features = getattr(self, method_name)(image_features, input_type=input_type, matryoshka_vis_token_scale=matryoshka_vis_token_scale)
                token_merging_outputs = None
            else:
                token_merging_outputs = getattr(self, method_name)(image_features, input_type=input_type)
                image_features = token_merging_outputs['image_features']

            image_features = image_features.to(self.get_model().mm_projector.weight.dtype)
            image_token_embeds = self.get_model().mm_projector(image_features)

        # only applied if `pe_type` in model config.`
        positional_encoding_outputs = self.positional_encoding(image_token_embeds, token_merging_outputs, input_type)

        projection_outputs = {
            "image_token_embeds": positional_encoding_outputs["image_token_embeds"],
            "position_ids": positional_encoding_outputs["position_ids"],
            "max_positions": positional_encoding_outputs["max_positions"],
        }
        return projection_outputs

    
    def project_v4(self, image_features, input_type="image", matryoshka_vis_token_scale=None):
        # wpq todo: add input_type="video"

        if matryoshka_vis_token_scale == '':
            return image_features

        H = W = int(self.get_model().get_vision_tower().config.image_size / self.get_model().get_vision_tower().config.patch_size)

        from rosemary import parse_kv_from_string
        kvs = parse_kv_from_string(matryoshka_vis_token_scale)

        if kvs['ver'] == 'v0':
            numtoks = int(kvs['numtoks'])
            B, H_W, D = image_features.shape
            reshaped_tensor = image_features.view(B, H, W, D)
            # (B, D, H, W)
            reshaped_tensor = reshaped_tensor.permute(0, 3, 1, 2)
            pool_size = stride = int( np.sqrt(H_W / numtoks) )
            # (B, D, 3, 3) if numtoks=9
            pooled_tensor = torch.nn.functional.avg_pool2d(reshaped_tensor, kernel_size=pool_size, stride=stride)
            image_features = pooled_tensor.permute(0, 2, 3, 1)
            # (B, numtoks, D)
            image_features = image_features.reshape(B, -1, D)
        else:
            raise ValueError(f"[ChatUniVi.model.arch] {kvs['ver']} not implemented.")

        return image_features


    def project_v3(self, image_features, input_type="image"):
        if input_type == "image":
            outputs = self.get_model().token_merge_image.merge_image_tokens(image_features)
        else:
            outputs = self.get_model().token_merging_video.merge_video_tokens(image_features)
        return outputs


    def project_v2(self, image_features, input_type="image"):
        if input_type == "image":
            outputs = self.get_model().token_merging_model.merge_image_tokens(image_features)
        else:
            outputs = self.get_model().token_merging_model.merge_video_tokens(image_features)
        return outputs


    def project_v1(self, image_features, input_type="image"):
        if input_type == "image":
            # [(B, 64, C), (B, 32, C), (B, 16, C)]
            cluster_image_features = []
            # assume square image. 576 -> 24
            s = int(math.sqrt(image_features.shape[1]))
            token_dict = {'x': image_features,
                            'token_num': image_features.size(1),
                            'idx_token': torch.arange(image_features.size(1))[None, :].repeat(
                                image_features.size(0), 1),
                            'agg_weight': image_features.new_ones(image_features.size(0), image_features.size(1),
                                                                1),
                            'mask': None,
                            'coord':  torch.stack(torch.meshgrid(
                                    torch.linspace(1/2, s-1/2, steps=s) / s,
                                    torch.linspace(1/2, s-1/2, steps=s) / s,
                                indexing='ij',
                            ), dim=-1).reshape(image_features.size(1), 2).repeat(image_features.size(0), 1, 1).to(image_features.device),
                            }

            token_dict = self.get_model().block0(self.get_model().ctm0(token_dict))
            cluster_image_features.append(token_dict["x"])

            token_dict = self.get_model().block1(self.get_model().ctm1(token_dict))
            cluster_image_features.append(token_dict["x"])

            token_dict = self.get_model().block2(self.get_model().ctm2(token_dict))
            cluster_image_features.append(token_dict["x"])

            # multiscale features: (B, 112, C)
            image_features = torch.cat(cluster_image_features, dim=1)
        else:
            # image_features.shape:
            # if input_type='video' then `image_features`: (38, 576, 1024) or (#frames, #patches, D)
            # use batch dimension to hold all the frames in a video.

            # cls_features: (1, 38, 1024) where num_frames=38
            cls_features = torch.mean(image_features, dim=1, keepdim=False).unsqueeze(0).clone()
            s = cls_features.shape[1]
            # cluster frames into events
            token_dict = {'x': cls_features,
                            'token_num': cls_features.size(1),
                            'idx_token': torch.arange(cls_features.size(1))[None, :].repeat(
                                cls_features.size(0), 1),
                            'agg_weight': cls_features.new_ones(cls_features.size(0), cls_features.size(1),
                                                                1),
                            'mask': None,
                            'coord': torch.stack(torch.meshgrid(
                                torch.linspace(1/2, s-1/2, steps=s) / s,
                            indexing='ij',
                        ), dim=-1).reshape(cls_features.size(1), 1).repeat(cls_features.size(0), 1, 1).to(cls_features.device),
                            }

            down_dict, token_dict = self.get_model().ctm3(token_dict)
            # print({k: v.shape if isinstance(v, torch.Tensor) else v for k, v in down_dict.items()})
            # {'x': torch.Size([1, 3, 1024]),  # 3 comes from 38 frames and 1/16 sampling rate -> 3 events
            #  'token_num': 3,
            #  'idx_token': torch.Size([1, 38]),
            #  'agg_weight': torch.Size([1, 38, 1]),
            #  'mask': None,
            #  'coord': torch.Size([1, 3, 1]),
            #  'index_down': torch.Size([1, 3]),
            #  'idx_cluster': torch.Size([1, 38])}

            # [[event0_frame0, event0_frame1, ...], [evenet1_frame0, ...], ...]
            events = collections.defaultdict(list)
            for id, i in enumerate(down_dict["idx_token"][0].tolist()):
                events[i].append(id)
            events = list(events.values())
            events = sorted(events, key=lambda x: x[0])

            cluster_image_features = []
            # assume square image. 576 -> 24
            s = int(math.sqrt(image_features.shape[1]))
            token_dict = {'x': image_features,
                        'token_num': image_features.size(1),
                        'idx_token': torch.arange(image_features.size(1))[None, :].repeat(
                            image_features.size(0), 1),
                        'agg_weight': image_features.new_ones(image_features.size(0), image_features.size(1),
                                                                1),
                        'mask': None,
                        'coord':  torch.stack(torch.meshgrid(
                                torch.linspace(1/2, s-1/2, steps=s) / s,
                                torch.linspace(1/2, s-1/2, steps=s) / s,
                            indexing='ij',
                        ), dim=-1).reshape(image_features.size(1), 2).repeat(image_features.size(0), 1, 1).to(image_features.device),
                        }

            token_dict0 = self.get_model().block0(self.get_model().ctm0(token_dict))
            token_dict1 = self.get_model().block1(self.get_model().ctm1(token_dict0))
            token_dict2 = self.get_model().block2(self.get_model().ctm2(token_dict1)) 


            # (#frames,) or (38,)
            # make spacing between patches across temporal dimension is same as that of xy dimension.
            num_frames_total = image_features.shape[0]
            coord_z = (1/2 + torch.arange(0, num_frames_total, 1)) / s

            for id, event_frame_ids in enumerate(events):
                
                ## layer0 

                # token_dict['x']: (38, 64, 1024) or (#frames, #clusters, D)
                # cur_image_features0: (1, #frames_in_event*#clusters, 1024)
                cur_image_features0 = torch.cat([token_dict0["x"][i] for i in event_frame_ids], dim=0).unsqueeze(0)
                # note here `token_dict0['x']` is used instead of `token_dict['x']` most likely due to computational reasons. if not use features fater `block0/ctm0`, then would need 576*64=36,864 patch features to represent 64 frames, each 576 tokens/frame. the distance matrix would be 5gb.
                # maybe can just do a 2x2 pool here, or just use level=-1 as visual representation.
                # this way, the distance matrix would cost ((336/14/2)**2*64)**2*4 / 1024/1024/1024=0.32gb to store.
                token_dict = {
                    'x': cur_image_features0,
                    'token_num': cur_image_features0.size(1),
                    'idx_token': torch.arange(cur_image_features0.size(1))[None, :].repeat(
                        cur_image_features0.size(0), 1),
                    'agg_weight': cur_image_features0.new_ones(cur_image_features0.size(0),
                                                                cur_image_features0.size(1),
                                                        1),
                    'mask': None,
                    # [(#clusters, 3) or (64, 3), ...] --cat--> (#cluster*#frames, 3)
                        'coord': torch.cat([
                            torch.cat((
                                token_dict0['coord'][idx_frame], # xy
                                token_dict0['coord'].new_ones((token_dict0['coord'].shape[1], 1)) * coord_z[idx_frame], # z
                            ), dim=-1)
                            for idx_frame in event_frame_ids
                        ], dim=0)
                    }

                cur_token_dict0 = self.get_model().block0(self.get_model().ctm0(token_dict))
                # {'x': torch.Size([1, 64, 1024]), 'token_num': 64, 'idx_token': torch.Size([1, 640]), 'agg_weight': torch.Size([1, 640, 1]), 'mask': None, 'coord': torch.Size([1, 64, 3]), 'index_down': torch.Size([1, 64]), 'idx_cluster': torch.Size([1, 640])}

                cluster_image_features.append(cur_token_dict0["x"])

                ## layer1
                cur_image_features1 = torch.cat([token_dict1["x"][i] for i in event_frame_ids], dim=0).unsqueeze(0)
                token_dict = {
                    'x': cur_image_features1,
                    'token_num': cur_image_features1.size(1),
                    'idx_token': torch.arange(cur_image_features1.size(1))[None, :].repeat(
                        cur_image_features1.size(0), 1),
                    'agg_weight': cur_image_features1.new_ones(cur_image_features1.size(0),
                                                                cur_image_features1.size(1),
                                                                1),
                    'mask': None,
                    # [(#clusters, 3) or (64, 3), ...] --cat--> (#cluster*#frames, 3)
                    'coord': torch.cat([
                        torch.cat((
                            token_dict1['coord'][idx_frame], # xy
                            token_dict1['coord'].new_ones((token_dict1['coord'].shape[1], 1)) * coord_z[idx_frame], # z
                        ), dim=-1)
                        for idx_frame in event_frame_ids
                    ], dim=0)
                    }

                cur_token_dict1 = self.get_model().block1(self.get_model().ctm1(token_dict))
                # {'x': torch.Size([1, 32, 1024]), 'token_num': 32, 'idx_token': torch.Size([1, 320]), 'agg_weight': torch.Size([1, 320, 1]), 'mask': None, 'coord': torch.Size([1, 32, 3]), 'index_down': torch.Size([1, 32]), 'idx_cluster': torch.Size([1, 320])}
                cluster_image_features.append(cur_token_dict1["x"])

                cur_image_features2 = torch.cat([token_dict2["x"][i] for i in event_frame_ids], dim=0).unsqueeze(0)
                token_dict = {'x': cur_image_features2,
                                'token_num': cur_image_features2.size(1),
                                'idx_token': torch.arange(cur_image_features2.size(1))[None, :].repeat(
                                    cur_image_features2.size(0), 1),
                                'agg_weight': cur_image_features2.new_ones(cur_image_features2.size(0),
                                                                            cur_image_features2.size(1),
                                                                            1),
                                'mask': None,
                            # [(#clusters, 3) or (64, 3), ...] --cat--> (#cluster*#frames, 3)
                            'coord': torch.cat([
                                torch.cat((
                                    token_dict2['coord'][idx_frame], # xy
                                    token_dict2['coord'].new_ones((token_dict2['coord'].shape[1], 1)) * coord_z[idx_frame], # z
                                ), dim=-1)
                                for idx_frame in event_frame_ids
                            ], dim=0)
                                }

                cur_token_dict2 = self.get_model().block2(self.get_model().ctm2(token_dict))
                cluster_image_features.append(cur_token_dict2["x"])

            # cat over 1. events 2. differnet levels within an event
            # `image_features`: (1, (64+32+16)*#frames, 1024)
            image_features = torch.cat(cluster_image_features, dim=1)

        return image_features

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, attention_mask, past_key_values, labels, images, matryoshka_vis_token_scale=None
    ):

        # images: 
        
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            if past_key_values is not None and vision_tower is not None and images is not None and input_ids.shape[1] == 1:
                attention_mask = torch.ones((attention_mask.shape[0], past_key_values[-1][-1].shape[-2] + 1), dtype=attention_mask.dtype, device=attention_mask.device)
            return input_ids, attention_mask, past_key_values, None, labels, None

        if type(images) is list or images.ndim == 5: # might is called during generation i think.
            concat_images = torch.cat([image for image in images], dim=0)
            image_features = self.encode_images(concat_images)
            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(image_features, split_sizes, dim=0)
            image_features = [x.flatten(0, 1) for x in image_features]
        else:
            image_features = self.encode_images(images)


        # print({
        #     'images.shape': images.shape, # (N_1+...+N_B , 3, H, W) where N_1 is 1 if "image" and number of frames if "video".
        #     '#images / example in batch': [(x==-200).sum().item() for x in input_ids],
        #     'image_features.shape': image_features.shape, # (N, #patches, D)
        # })
        # {'images.shape': torch.Size([38, 3, 336, 336]), 
        #  '#images / example in batch': [23, 1, 14], 
        #  'image_features.shape': torch.Size([38, 576, 1024])}


        new_position_ids = []
        new_input_embeds = []
        new_labels = [] if labels is not None else None
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            if (cur_input_ids == IMAGE_TOKEN_INDEX).sum() == 0:
                # multimodal LLM, but the current sample is not multimodal
                cur_input_embeds = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = cur_input_embeds + (
                            0. * self.get_model().mm_projector(vision_tower.dummy_feature)).sum()
                new_input_embeds.append(cur_input_embeds)
                if labels is not None:
                    new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            cur_position_ids_start = 0

            # image_token_indices: tensor([35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48], device='cuda:0')
            image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]

            cur_new_input_embeds = []
            cur_new_position_ids = [] # wpq: position_id for current example in the batch
            if labels is not None:
                cur_labels = labels[batch_idx]
                cur_new_labels = []
                assert cur_labels.shape == cur_input_ids.shape

            if len(image_token_indices) > 1:

                ## `temp` groups `image_token_indices` into consecutive spans
                # e.g., [1,2,5,6,7] -> [[1,2], [5,6,7]]
                # [[tensor(35, device='cuda:0'), tensor(36, device='cuda:0'), tensor(37, device='cuda:0'), tensor(38, device='cuda:0'), tensor(39, device='cuda:0'), tensor(40, device='cuda:0'), tensor(41, device='cuda:0'), tensor(42, device='cuda:0'), tensor(43, device='cuda:0'), tensor(44, device='cuda:0'), tensor(45, device='cuda:0'), tensor(46, device='cuda:0'), tensor(47, device='cuda:0'), tensor(48, device='cuda:0')]]
                temp = []
                cur, pre = image_token_indices[0], image_token_indices[0]
                for i in image_token_indices:
                    cur = i
                    if cur - pre == 1:
                        temp[-1] = temp[-1] + [cur]
                    else:
                        temp.append([cur])
                    pre = cur

                if len(temp) != 1:
                    raise ValueError(f'should contain 1 consecutive <image> tags but got {len(temp)}: {temp}')

                for i in temp:
                    image_token_start = image_token_indices[0] # tensor(35, device='cuda:0')
                    image_token_end = image_token_indices[-1]  # tensor(57, device='cuda:0')
                    cur_image_features = []

                    for _ in i:
                        cur_image_features.append(image_features[cur_image_idx])
                        cur_image_idx += 1

                    # cur_image_features: [(576, 1024), ..., (576, 1024)]. features for each frame in the video.
                    cur_image_features = torch.stack(cur_image_features, dim=0)
                    # cur_image_features: (#frames, 576, 1024)
                    projection_outputs = self.project(cur_image_features, input_type="video", matryoshka_vis_token_scale=matryoshka_vis_token_scale)
                    cur_image_features = projection_outputs["image_token_embeds"]
                    # cur_image_features: (1, 64+32+16, 1024)
                    t, l, n = cur_image_features.size()
                    cur_image_features = cur_image_features.contiguous().view(t * l, n)

                    cur_position_ids = projection_outputs['position_ids'].squeeze(0)
                    cur_image_token_embeds_max_positions = projection_outputs['max_positions']

                    if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
                        cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[:image_token_start - 1]).detach())
                        cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[image_token_start - 1:image_token_start]))
                        cur_new_input_embeds.append(cur_image_features)
                        cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[image_token_end + 1:image_token_end + 2]))
                        if labels is not None:
                            cur_new_labels.append(cur_labels[:image_token_start])
                            cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=labels.device, dtype=labels.dtype))
                            cur_new_labels.append(cur_labels[image_token_end:image_token_end + 1])
                            cur_labels = cur_labels[image_token_end + 2:]
                        cur_new_position_ids.append(torch.arange(cur_position_ids_start, cur_position_ids_start+image_token_start, device=input_ids.device, dtype=torch.long))
                        cur_new_position_ids.append(cur_position_ids_start + image_token_start + cur_position_ids + 1)
                        cur_position_ids_start += image_token_start + cur_image_token_embeds_max_positions + 1
                    else:
                        # [(#tokens, D), ...]
                        cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[:image_token_start]))
                        cur_new_input_embeds.append(cur_image_features)
                        if labels is not None:
                            cur_new_labels.append(cur_labels[:image_token_start])  # [(#tokens,), ...]
                            # append label for visual tokens of size (#tokens, ) with value -100
                            cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=labels.device, dtype=labels.dtype))
                            cur_labels = cur_labels[image_token_end+1:]
                        cur_new_position_ids.append(torch.arange(cur_position_ids_start, cur_position_ids_start+image_token_start, device=input_ids.device, dtype=torch.long))
                        cur_new_position_ids.append(cur_position_ids_start + image_token_start + cur_position_ids)
                        cur_position_ids_start += image_token_start + cur_image_token_embeds_max_positions # skip <image> token

                if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end',
                                                                                  False):
                    cur_input_ids = cur_input_ids[image_token_end + 2:]
                else:
                    cur_input_ids = cur_input_ids[image_token_end + 1:]

            elif image_token_indices.numel() > 0: 
                # wpq: preprocess images.
                cur_image_features = []
                image_token_start = image_token_indices[0]
                image_token_end = image_token_indices[-1]

                for _ in image_token_indices:
                    cur_image_features.append(image_features[cur_image_idx])
                    cur_image_idx += 1

                # (B, N, D)
                cur_image_features = torch.stack(cur_image_features, dim=0)
                # (B, 64+32+16=112, D)
                projection_outputs = self.project(cur_image_features, input_type="image", matryoshka_vis_token_scale=matryoshka_vis_token_scale)
                cur_image_features = projection_outputs["image_token_embeds"]
                t, l, n = cur_image_features.size()
                # (B*112, D)
                cur_image_features = cur_image_features.contiguous().view(t * l, n)
                cur_position_ids = projection_outputs['position_ids'].squeeze(0)
                cur_image_token_embeds_max_positions = projection_outputs['max_positions']


                if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[:image_token_start-1]).detach())
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[image_token_start-1:image_token_start]))
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[image_token_end+1:image_token_end+2]))
                    if labels is not None:
                        cur_new_labels.append(cur_labels[:image_token_start])
                        cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=labels.device, dtype=labels.dtype))
                        cur_new_labels.append(cur_labels[image_token_end:image_token_end+1])
                        cur_labels = cur_labels[image_token_end+2:]
                    cur_new_position_ids.append(torch.arange(cur_position_ids_start, cur_position_ids_start+image_token_start, device=input_ids.device, dtype=torch.long))
                    cur_new_position_ids.append(cur_position_ids_start + image_token_start + cur_position_ids)
                    cur_position_ids_start += image_token_start + cur_image_token_embeds_max_positions + 1
                else:
                    # this branch since `tune_mm_mlp_adapter` typically set to False
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[:image_token_start]))
                    cur_new_input_embeds.append(cur_image_features)
                    if labels is not None:
                        cur_new_labels.append(cur_labels[:image_token_start])
                        cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=labels.device, dtype=labels.dtype))
                        cur_labels = cur_labels[image_token_end+1:]
                    cur_new_position_ids.append(torch.arange(cur_position_ids_start, cur_position_ids_start+image_token_start, device=input_ids.device, dtype=torch.long))
                    cur_new_position_ids.append(cur_position_ids_start + image_token_start + cur_position_ids)
                    cur_position_ids_start += image_token_start + cur_image_token_embeds_max_positions

                if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
                    cur_input_ids = cur_input_ids[image_token_end+2:]
                else:
                    # this branch since `tune_mm_mlp_adapter` typically set to False
                    cur_input_ids = cur_input_ids[image_token_end+1:]

            ## shared by both image & video to append the latter part of the input_ids/labels.
            if cur_input_ids.numel() > 0:
                if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids).detach())
                else:
                    # this branch since `tune_mm_mlp_adapter` typically set to False
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids))
                if labels is not None:
                    cur_new_labels.append(cur_labels)
                cur_new_position_ids.append(torch.arange(cur_position_ids_start, cur_position_ids_start + cur_input_ids.shape[0], device=input_ids.device, dtype=torch.long))

            cur_new_input_embeds = [x.to(device=self.device) for x in cur_new_input_embeds]
            # (seq_len, D)
            cur_new_input_embeds = torch.cat(cur_new_input_embeds, dim=0)
            new_input_embeds.append(cur_new_input_embeds)
            if labels is not None:
                cur_new_labels = torch.cat(cur_new_labels, dim=0)
                new_labels.append(cur_new_labels)
            # (seq_len,)
            cur_new_position_ids = torch.cat(cur_new_position_ids, dim=0)
            new_position_ids.append(cur_new_position_ids)


        # does padding to longest sequence.
        # does not take into account of pad left/right. just implements pad right.
        # therefore, only works for batched training & batch_size=1 inference.
        if any(x.shape != new_input_embeds[0].shape for x in new_input_embeds):
            max_len = max(x.shape[0] for x in new_input_embeds)

            new_input_embeds_align = []
            for cur_new_embed in new_input_embeds:
                # [(#token_b=1, D), ..., (#tokens_b=B, D)] -> [(max_len, D), ..., (max_len, D)]
                cur_new_embed = torch.cat((cur_new_embed, torch.zeros((max_len - cur_new_embed.shape[0], cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)), dim=0)
                new_input_embeds_align.append(cur_new_embed)
            # (B, max_seq, D)
            new_input_embeds = torch.stack(new_input_embeds_align, dim=0)

            if labels is not None:
                # wpq: padding left/right not done correctly.
                # but since `labels` only supplied during training, always pad right. therefore runs ok.
                new_labels_align = []
                _new_labels = new_labels
                for cur_new_label in new_labels:
                    cur_new_label = torch.cat((cur_new_label, torch.full((max_len - cur_new_label.shape[0],), IGNORE_INDEX, dtype=cur_new_label.dtype, device=cur_new_label.device)), dim=0)
                    new_labels_align.append(cur_new_label)
                new_labels = torch.stack(new_labels_align, dim=0)

            if new_position_ids is not None:
                # pad 0 to position_ids on the right if sequence is too short.
                new_position_ids_align = []
                for cur_new_position_ids in new_position_ids:
                    cur_new_position_ids = torch.cat((cur_new_position_ids, torch.full((max_len - cur_new_position_ids.shape[0],), 0, dtype=cur_new_position_ids.dtype, device=cur_new_position_ids.device)), dim=0)
                    new_position_ids_align.append(cur_new_position_ids)
                new_position_ids = torch.stack(new_position_ids_align, dim=0)
                assert(new_position_ids.shape == new_input_embeds.shape[:2])

            if attention_mask is not None:
                new_attention_mask = []
                # `attention_mask`: padded to max_seq before `input_embed` is expanded to replace <image>
                # `labels`: padded to max_seq before `labels` is expanded to replace <image>
                # `cur_new_labels`: token embeds that expands <image> to tokens.
                # `cur_new_labels_align`: token embeds that expands <image> to tokens padded to max_seq in batch
                for cur_attention_mask, cur_new_labels, cur_new_labels_align in zip(attention_mask, _new_labels, new_labels):
                    # for SFT, we know that image is always attended to, therefore can just prepend 
                    # [True, ..., True] to the attention mask.
                    #         mask toks from <image>       original attn mask       pad to max_seq
                    # concat: [True, ..., True]         [True, True, ..., False] [False, ..., False]
                    new_attn_mask_pad_left = torch.full((cur_new_labels.shape[0] - labels.shape[1],), True, dtype=attention_mask.dtype, device=attention_mask.device)
                    new_attn_mask_pad_right = torch.full((cur_new_labels_align.shape[0] - cur_new_labels.shape[0],), False, dtype=attention_mask.dtype, device=attention_mask.device)
                    cur_new_attention_mask = torch.cat((new_attn_mask_pad_left, cur_attention_mask, new_attn_mask_pad_right), dim=0)
                    new_attention_mask.append(cur_new_attention_mask)
                attention_mask = torch.stack(new_attention_mask, dim=0)

                if attention_mask.shape != new_labels.shape:
                    print('[prepare_inputs_labels_for_multimodal] attention_mask.shape != new_labels.shape or {attention_mask.shape} != {new_labels.shape}. infer attention_mask from input_ids')
                    print('input_ids: ', input_ids)
                    # basically attend to all of (input, response). specifically, assume pad_id=0, therefore just find input_id that is not padded and assign True to corresponding location in attention_mask. 
                    new_attention_mask = []
                    for cur_input_ids, cur_attention_mask, cur_new_labels, cur_new_labels_align in zip(input_ids, attention_mask, _new_labels, new_labels):
                        new_attn_mask_pad_left = torch.full((cur_new_labels.shape[0] - labels.shape[1],), True, dtype=attention_mask.dtype, device=attention_mask.device)
                        new_attn_mask_pad_right = torch.full((cur_new_labels_align.shape[0] - cur_new_labels.shape[0],), False, dtype=attention_mask.dtype, device=attention_mask.device)
                        cur_new_attention_mask = torch.cat((new_attn_mask_pad_left, (cur_input_ids!=0), new_attn_mask_pad_right), dim=0)
                        new_attention_mask.append(cur_new_attention_mask)
                    attention_mask = torch.stack(new_attention_mask, dim=0)

        else:
            new_input_embeds = torch.stack(new_input_embeds, dim=0)
            if labels is not None:
                new_labels = torch.stack(new_labels, dim=0)

            if new_position_ids is not None:
                new_position_ids = torch.stack(new_position_ids, dim=0)

            if attention_mask is not None:
                new_attn_mask_pad_left = torch.full((attention_mask.shape[0], max(new_input_embeds.shape[1] - attention_mask.shape[1], 0)), True, dtype=attention_mask.dtype, device=attention_mask.device)
                attention_mask = torch.cat((new_attn_mask_pad_left, attention_mask), dim=1)
                assert(attention_mask.shape == new_input_embeds.shape[:2])


        return None, attention_mask, past_key_values, new_input_embeds, new_labels, new_position_ids

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            tokenizer.add_tokens([DEFAULT_VIDEO_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_VID_START_TOKEN, DEFAULT_VID_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = Fal
                    
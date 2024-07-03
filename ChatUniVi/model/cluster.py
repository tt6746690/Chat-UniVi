import torch
import math
import torch.nn as nn
import warnings
import collections


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    try:
        return _no_grad_trunc_normal_(tensor, mean, std, a, b)
    except:
        return tensor


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


def index_points(points, idx):
    """Sample features following the index.
    Returns:
        new_points:, indexed points data, [B, S, C]

    Args:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx.to(device), :]
    return new_points


def cluster_dpc_knn(token_dict, cluster_num, k=5, token_mask=None, token_coord=None, coord_weight=0):
    """Cluster tokens with DPC-KNN algorithm.
    Return:
        idx_cluster (Tensor[B, N]): cluster index of each token.
        cluster_num (int): actual cluster number. The same with
            input cluster number
    Args:
        token_dict (dict): dict for token information
        cluster_num (int): cluster number
        k (int): number of the nearest neighbor used for local density.
        token_mask (Tensor[B, N]): mask indicate the whether the token is
            padded empty token. Non-zero value means the token is meaningful,
            zero value means the token is an empty token. If set to None, all
            tokens are regarded as meaningful.
    """
    with torch.no_grad():
        x = token_dict["x"]
        B, N, C = x.shape

        # (B, N, N)
        dist_matrix = torch.cdist(x.float(), x.float()) / (C ** 0.5)

        if token_mask is not None:
            token_mask = token_mask > 0
            # in order to not affect the local density, the distance between empty tokens
            # and any other tokens should be the maximal distance.
            dist_matrix = dist_matrix * token_mask[:, None, :] + \
                          (dist_matrix.max() + 1) * (~token_mask[:, None, :])
        
        # wpq: distance in xy coordinate space to encourage merging of nearby tokens.
        if token_coord is not None and coord_weight > 0:
            coord_dim = token_coord.shape[-1]
            dist_matrix_coord = torch.cdist(token_coord.float(), token_coord.float()) / math.sqrt(coord_dim)
            dist_matrix = dist_matrix * torch.exp(coord_weight*dist_matrix_coord)

        # get local density

        dist_nearest, index_nearest = torch.topk(dist_matrix, k=k, dim=-1, largest=False)
        # (B, N)
        density = (-(dist_nearest ** 2).mean(dim=-1)).exp()
        # add a little noise to ensure no tokens have the same density.
        density = density + torch.rand(
            density.shape, device=density.device, dtype=density.dtype) * 1e-6

        if token_mask is not None:
            # the density of empty token should be 0
            density = density * token_mask

        # get distance indicator
        mask = density[:, None, :] > density[:, :, None]
        mask = mask.type(x.dtype)
        # wpq: https://github.com/PKU-YuanGroup/Chat-UniVi/issues/40
        # previously, flatten gets max distance of each batch, but we want max distance for each token.
        # in practice, these two max distance does not vary by that much, e.g., max(batch)=7.486 vs. max(token)\in [5.7, 7.48]
        # in addition, after masking (next 2 lines), the resulting `dist` is identical for 1 image I tested due the presence of a single cluster that has very high score.
        # dist_max = dist_matrix.flatten(1).max(dim=-1)[0][:, None, None]
        dist_max = dist_matrix.max(dim=-1)[0][:, :, None]
        # (B, N)
        dist, index_parent = (dist_matrix * mask + dist_max * (1 - mask)).min(dim=-1)

        # select clustering center according to score
        # (B, N)
        score = dist * density

        # index_down (B, k) having values in [0,...,N-1] representing the cluster centers
        # e.g., contains the token index that represents the cluster centers.
        _, index_down = torch.topk(score, k=cluster_num, dim=-1)

        # assign tokens to the nearest center
        dist_matrix = index_points(dist_matrix, index_down)
        # (B, N) having values [0,...,64]
        idx_cluster = dist_matrix.argmin(dim=1)

        # make sure cluster center merge to itself
        # wpq: really a no-op 
        # (B, k) where idx_batch[i] = [i, ..., i]
        idx_batch = torch.arange(B, device=x.device)[:, None].expand(B, cluster_num)
        # (B, k) where idx_batch[i] = [0, ..., k]
        idx_tmp = torch.arange(cluster_num, device=x.device)[None, :].expand(B, cluster_num)
        idx_cluster[idx_batch.reshape(-1), index_down.reshape(-1)] = idx_tmp.reshape(-1)

    return idx_cluster, cluster_num, index_down


def merge_tokens(token_dict, idx_cluster, cluster_num, token_weight=None):
    """Merge tokens in the same cluster to a single cluster.
    Implemented by torch.index_add(). Flops: B*N*(C+2)
    Return:
        out_dict (dict): dict for output token information

    Args:
        token_dict (dict): dict for input token information
        idx_cluster (Tensor[B, N]): cluster index of each token.
        cluster_num (int): cluster number
        token_weight (Tensor[B, N, 1]): weight for each token.
    """

    x = token_dict['x']
    idx_token = token_dict['idx_token']
    agg_weight = token_dict['agg_weight']

    B, N, C = x.shape
    if token_weight is None:
        token_weight = x.new_ones(B, N, 1)

    idx_batch = torch.arange(B, device=x.device)[:, None]
    # (B, N): make cluster_id of each example in a batch unique.
    idx = idx_cluster + idx_batch * cluster_num

    # (B*N, 1)
    # for each example in batch, for each cluster center: add weights of tokens that is in the cluster.
    # if uniform weight of 1 per token, this is equivalent to counting the number of examples within each cluster!
    # assert(
    #     torch.isclose(idx_cluster.squeeze().unique(return_counts=True)[1].reshape(all_weight.shape).to(all_weight.dtype), all_weight).all().item()
    # )
    all_weight = token_weight.new_zeros(B * cluster_num, 1)
    all_weight.index_add_(dim=0, index=idx.reshape(B * N),
                          source=token_weight.reshape(B * N, 1))
    all_weight = all_weight + 1e-6
    # all_weight: (B*N, 1)
    # idx: (B, N)
    # norm_weight: (B, N, 1) per-token weight, token from a large cluster is assigned a smaller weight
    norm_weight = token_weight / all_weight[idx]

    # average token features
    # (B*k, C): token from a large cluster is assigned a smaller weight, i.e., vectorized average.
    x_merged = x.new_zeros(B * cluster_num, C)
    source = x * norm_weight
    x_merged.index_add_(dim=0, index=idx.reshape(B * N),
                        source=source.reshape(B * N, C).type(x.dtype))
    x_merged = x_merged.reshape(B, cluster_num, C)

    # average spatial coordinate
    if 'coord' in token_dict:
        coord = token_dict['coord']
        coord_dim = coord.shape[-1] # 1, 2, or 3
        coord_merged = coord.new_zeros(B * cluster_num, coord_dim)
        source = coord * norm_weight
        coord_merged.index_add_(dim=0, index=idx.reshape(B * N),
                                source=source.reshape(B * N, coord_dim).type(coord.dtype))
        coord_merged = coord_merged.reshape(B, cluster_num, coord_dim)

    # (B, N)
    idx_token_new = index_points(idx_cluster[..., None], idx_token).squeeze(-1)
    # (B, N, 1)
    weight_t = index_points(norm_weight, idx_token)
    agg_weight_new = agg_weight * weight_t
    agg_weight_new / agg_weight_new.max(dim=1, keepdim=True)[0]

    out_dict = {}
    out_dict['x'] = x_merged
    out_dict['token_num'] = cluster_num
    out_dict['idx_token'] = idx_token_new
    out_dict['agg_weight'] = agg_weight_new
    out_dict['mask'] = None
    if 'coord' in token_dict:
        out_dict['coord'] = coord_merged
    return out_dict


def order_tokens(token_dict, token_ordering="default"):
    """sort cluster id according to some criterion.
    e.g., if `token_ordering` is "raster", then smaller cluster idx has small (x,y,z) coord.
    """

    if token_ordering == "default":
        return token_dict
    elif token_ordering in ("raster", "random", "clustersize"):
        # x, coord
        coord = token_dict["coord"]
        x = token_dict["x"]
        idx_token = token_dict["idx_token"]
        # `agg_weight`: assigned to the input tokens, agnostic of cluster id so don't need to change anything.

        # sort_score: (B, N)
        if token_ordering == "random":
            sort_score = torch.rand_like(coord[...,0])
        elif token_ordering == "raster":
            # maps multi-dimensional coordinate [z,y,x] -> z*100+10*y+x for sorting.
            sort_score = torch.stack(
                [(10 ** (coord.shape[-1] - di - 1)) * coord[..., di] for di in range(coord.shape[-1])], dim=-1
            ).sum(dim=-1)
            # maps multi-dimensional coordinate [x,y,z] -> x+y*10+z*100 for sorting. wrong ordering!
            # sort_score = torch.stack([(10**(coord.shape[-1]-di))*coord[...,di] for di in range(coord.shape[-1])], dim=-1).sum(dim=-1)
        elif token_ordering == 'clustersize':
            # larger cluster size earlier in the ordering
            cluster_sizes = torch.stack([i.unique(return_counts=True)[1] for i in idx_token])
            sort_score = -cluster_sizes
        # [new_cluster_id: old_cluster_id]
        inds = torch.argsort(sort_score, dim=-1, descending=False)
        # [old_cluster_id: new_cluster_id]
        inds_reverse = torch.argsort(inds, dim=-1)

        token_dict["x"] = torch.take_along_dim(x, inds[..., None].repeat(1, 1, x.shape[-1]), dim=1).reshape_as(x)
        token_dict["coord"] = torch.take_along_dim(
            coord, inds[..., None].repeat(1, 1, coord.shape[-1]), dim=1
        ).reshape_as(coord)
        token_dict["idx_token"] = index_points(inds_reverse[..., None], idx_token).reshape_as(idx_token)

        return token_dict
    else:
        raise ValueError(f"Invalid token_ordering={token_ordering}")


def take_firstn_clusters(token_dict, n=None):
    """Takes first `n` cluster in `token_dict`. """
    if n is None:
        return token_dict

    out_dict = {}
    out_dict['x'] = token_dict['x'][:, :n, :]
    out_dict['token_num'] = n
    idx_token = token_dict['idx_token']
    # random assign clusters that are too small.
    mask = idx_token >= n
    idx_token[mask] = torch.randint(0, n, size=(torch.sum(mask),), device=idx_token.device)
    out_dict['idx_token'] = idx_token
    out_dict['agg_weight'] = token_dict['agg_weight'] # not used anywhere
    out_dict['mask'] = token_dict['mask']
    out_dict['coord'] = token_dict['coord'][:, :n, :]

    return out_dict



class CTM(nn.Module):
    def __init__(self, sample_ratio, embed_dim, dim_out, k=5, coord_weight=0, token_ordering="default", prune_ratio=None):
        super().__init__()
        self.sample_ratio = sample_ratio
        self.dim_out = dim_out
        self.k = k
        self.coord_weight = coord_weight
        self.token_ordering = token_ordering
        self.prune_ratio = prune_ratio

    def forward(self, token_dict, sample_ratio=None):

        x = token_dict["x"]
        B, N, C = x.shape

        token_weight = x.new_ones(B, N)

        if token_dict["mask"] is not None:
            token_weight.masked_fill_((1 - token_dict["mask"]).to(torch.bool), float("-inf"))
        token_weight = token_weight.unsqueeze(2)
        token_dict["x"] = x

        if sample_ratio is not None:
            cluster_num = max(math.ceil(N * sample_ratio), 1)  # (t,)
        elif self.sample_ratio > 1:
            cluster_num = max(math.ceil(self.sample_ratio), 1)  # (h,w)
        else:
            cluster_num = max(math.ceil(N * self.sample_ratio), 1)  # (t,h,w)

        if self.prune_ratio is None or self.prune_ratio <= 0:
            prune_num = 0
        else:
            if self.prune_ratio >= 1:
                prune_num = max(math.ceil(self.prune_ratio), 1)
            else:
                prune_num = max(math.ceil(N * self.prune_ratio), 1)
        # if event consists of 1 frame,  N=64, cluster_num=64, prune_num>0
        # then would be trying to cluster (64+prune_num) given 64 data points. 
        # in this case, just don't do any pruning.
        if cluster_num+prune_num > N:
            prune_num = 0
        
        k = min(3, max(cluster_num // 2, 1)) if self.k > cluster_num else self.k

        token_coord = token_dict["coord"] if "coord" in token_dict else None
        idx_cluster, _, _ = cluster_dpc_knn(
            token_dict,
            cluster_num+prune_num,
            k,
            token_mask=token_dict["mask"],
            token_coord=token_coord,
            coord_weight=self.coord_weight,
        )

        down_dict = merge_tokens(token_dict, idx_cluster, cluster_num+prune_num, token_weight)

        if prune_num != 0:
            # note if prune clusters, will undermine `token_ordering="default"`
            # in this case, `token_ordering="default"` is equivalent to `token_ordering="clustersize"`
            down_dict = order_tokens(down_dict, token_ordering="clustersize")
            down_dict = take_firstn_clusters(down_dict, n=cluster_num)

        down_dict = order_tokens(down_dict, self.token_ordering)

        return down_dict, token_dict


class TCBlock(nn.Module):
    
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        sr_ratio=1,
        use_sr_layer=False,
    ):
        super().__init__()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, inputs):
        if isinstance(inputs, tuple) or isinstance(inputs, list):
            q_dict, kv_dict = inputs
        else:
            q_dict, kv_dict = inputs, None
        return q_dict


def create_token_dict_from_features(image_features, sizes=None):
    """Create `token_dict` as input to clustering layers.
    If `sizes` is None, then `image_features` assumed to be 1d signal.
    """

    B, N, D = image_features.shape
    coord_dim = len(sizes)
    # (N, d)
    coord = torch.stack(
        torch.meshgrid(
            *[
                # coordinate normalized to [0,1]
                ((1 / 2 + torch.arange(s)) / s)
                for s in sizes
            ],
            indexing="ij",
        ),
        dim=-1,
    )

    # (B, N, d)
    coord = coord.reshape(N, coord_dim).repeat(B, 1, 1).to(image_features.device)
    # (B, N)
    idx_token = torch.arange(N)[None, :].repeat(B, 1)
    # (B, N, 1)
    agg_weight = image_features.new_ones(B, N, 1)

    token_dict = {
        "x": image_features,
        "token_num": N,
        "idx_token": idx_token,
        "agg_weight": agg_weight,
        "mask": None,
        "coord": coord,
    }

    return token_dict


class TokenMergeClusterDPCKNN(nn.Module):
    """Token merging with dpc-knn. Agnostic of dimension of features (e.g., 1d, image, video).
    but need to supply `self.forward` with properly constructed `token_dict`."""

    def __init__(self, sample_ratios, ks, coord_weight, token_ordering, prune_ratios=None):
        super().__init__()
        self.sample_ratios = sample_ratios
        self.ks = ks
        self.coord_weight = coord_weight
        self.token_ordering = token_ordering
        self.prune_ratios = prune_ratios if prune_ratios else [None]*len(sample_ratios)

        # not really used anywhere
        embed_dim = 1024
        dim_out = 1024

        self.ctms = nn.ModuleList()
        self.tc_blocks = nn.ModuleList()

        for sample_ratio, k, prune_ratio in zip(sample_ratios, ks, self.prune_ratios):
            ctm = CTM(
                sample_ratio=sample_ratio,
                embed_dim=embed_dim,
                dim_out=dim_out,
                coord_weight=coord_weight,
                k=k,
                token_ordering=token_ordering,
                prune_ratio=prune_ratio,
            )
            block = TCBlock(dim=dim_out, num_heads=8)
            self.ctms.append(ctm)
            self.tc_blocks.append(block)

    def forward(self, token_dict):
        """Returns [token_list, token_list0, ...,]
        a list of state, e.g., cluster membership, merged features etc. initially and after each layer.
        """
        token_dict_list = [token_dict]
        for ctm, block in zip(self.ctms, self.tc_blocks):
            # x: (B, N, D) -> (B, #clusters, D)
            # coord: (B, N, coord_dim) -> (B, #clusters, coord_dim)
            token_dict = block(ctm(token_dict))
            token_dict_list.append(token_dict)
        return token_dict_list

    def __repr__(self):
        return "".join(
            [
                "TokenMergeClusterDPCKNN",
                "(",
                str(self.sample_ratios).replace(" ", "") + ",",
                str(self.ks).replace(" ", "") + ",",
                str(self.coord_weight) + ",",
                str(self.token_ordering) + ",",
                str(self.prune_ratios),
                ")",
            ]
        )


class VideoTokenMergeClusterDPCKNN(nn.Module):

    def __init__(
        self,
        sample_ratios_temporal,
        sample_ratios_spatial,
        sample_ratios_video,
        ks,
        coord_weights,
        token_orderings,
        prune_ratios_spatial,
        prune_ratios_video,
    ):
        super().__init__()
        coord_weight_temporal, coord_weight_spatial, coord_weight_video = coord_weights
        token_ordering_temporal, token_ordering_spatial, token_ordering_video = token_orderings
        prune_ratios_spatial = prune_ratios_spatial if prune_ratios_spatial else [None]*len(sample_ratios_spatial)
        prune_ratios_video = prune_ratios_video if prune_ratios_video else [None]*len(sample_ratios_video)

        self.token_merge_temporal = TokenMergeClusterDPCKNN(
            sample_ratios=sample_ratios_temporal,
            ks=ks[:1],
            coord_weight=coord_weight_temporal,
            token_ordering=token_ordering_temporal,
        )
        self.token_merge_image = TokenMergeClusterDPCKNN(
            sample_ratios=sample_ratios_spatial,
            ks=ks,
            coord_weight=coord_weight_spatial,
            token_ordering=token_ordering_spatial,
            prune_ratios=prune_ratios_spatial,
        )
        self.token_merge_video_list = torch.nn.ModuleList(
            [
                TokenMergeClusterDPCKNN(
                    sample_ratios=[sample_ratio],
                    ks=[k],
                    coord_weight=coord_weight_video,
                    token_ordering=token_ordering_video,
                    prune_ratios=[prune_ratio]
                )
                for sample_ratio, k, prune_ratio in zip(
                        sample_ratios_video,
                        ks,
                        prune_ratios_video,
                    )
            ]
        )

    def forward(self, image_features):

        # `image_features`: (#frames, #patches, D)
        _, P, D = image_features.shape
        device = image_features.device

        # token merging on each frame's global avg pool features
        # cls_features: (1, #frames, D)
        cls_features = torch.mean(image_features, dim=1, keepdim=False).unsqueeze(0).clone()
        token_dict_temporal = create_token_dict_from_features(
            cls_features, (cls_features.shape[1],))
        token_dict_temporal = self.token_merge_temporal(token_dict_temporal)[-1]

        # [[event0_frame0, event0_frame1, ...], [event1_frame0, ...], ...]
        events = collections.defaultdict(list)
        for id, i in enumerate(token_dict_temporal["idx_token"][0].tolist()):
            events[i].append(id)
        # sort by cluster id ascending
        events = list(zip(*sorted(events.items(), key=lambda x: x[0])))[1]

        # token merging applied to each frame's patches independently.
        # this is most likely to reduce the number of tokens before token merging in 3d.
        # featues: (#frames, #patches, D) -> (#frames, #clusters, D)
        # or (64, 576, 1024) -> (64, 64, 1024) -> (64, 32, 1024) -> (64, 16, 1024)
        sizes = (int(math.sqrt(image_features.shape[1])),) * 2
        token_dict = create_token_dict_from_features(image_features, sizes)
        token_dict_image_list = self.token_merge_image(token_dict)[1:]

        # t-dimension spacing is same as xy-dimension spacing.
        num_frames_total = image_features.shape[0]
        side_len = int(math.sqrt(image_features.shape[1]))
        coord_z = (1 / 2 + torch.arange(0, num_frames_total, 1, device=device)) / side_len

        ## iterate over events
        token_dict_video_list = []
        for event_frame_ids in events:

            # number of patches in an event
            num_frames = len(event_frame_ids)
            Np_event = num_frames * P

            ## iterate over token merging layers on video segments/events.
            token_dict_event_list = []
            for level in range(len(self.token_merge_video_list)):
                token_merge_video = self.token_merge_video_list[level]
                token_dict_frames = token_dict_image_list[level]

                # number of clusters in a frame after per-frame token merging
                Nc_frame = self.token_merge_image.sample_ratios[level]
                # number of clusters in an event after per-frame token merging
                N = num_frames * Nc_frame

                # create token dict from result of frame-wise token merging
                # (1, #frames*#clusters, D)
                event_features = token_dict_frames["x"][event_frame_ids].reshape(N, D).unsqueeze(0)
                # (1, #frames_in_event*#patches) convert per-frame index to per-event index, e.g., i-th frame's index added by i*#clusters
                event_idx_token = token_dict_frames["idx_token"][event_frame_ids].reshape(Np_event)
                event_idx_token = (
                    torch.arange(num_frames, device=device).repeat_interleave(P) * Nc_frame + event_idx_token
                )
                event_idx_token = event_idx_token.unsqueeze(0)
                # (1, #frames_in_event*#patches, 1)
                event_agg_weight = token_dict_frames["agg_weight"][event_frame_ids].reshape(1, Np_event, 1)
                # (1, #frames*#clusters, 3)
                coord = torch.cat(
                    (
                        coord_z[event_frame_ids][..., None, None].repeat(1, Nc_frame, 1),  # (#frames, #clusters, 1)
                        token_dict_frames["coord"][event_frame_ids],  # (#frames, #clusters, 2)
                    ),
                    dim=-1,
                ).reshape(1, N, 3)

                token_dict = {
                    "x": event_features,
                    "token_num": N,
                    "idx_token": event_idx_token,
                    "agg_weight": event_agg_weight,
                    "mask": None,
                    "coord": coord,
                }

                token_dict_video = token_merge_video(token_dict)[-1]
                token_dict_event_list.append(token_dict_video)
            token_dict_video_list.append(token_dict_event_list)

        video_features = []
        for token_dict_event_list in token_dict_video_list:
            # (1, 64+32+16, 1024)
            event_features = torch.cat([d["x"] for d in token_dict_event_list], dim=1)
            video_features.append(event_features)
        # (1, (64+32+16)*#frames, 1024)
        video_features = torch.cat(video_features, dim=1)

        outputs = {
            "events": events,
            "video_features": video_features,
            "token_dict_temporal": token_dict_temporal,
            "token_dict_image_list": token_dict_image_list,
            "token_dict_video_list": token_dict_video_list,
        }

        return outputs

from abc import ABC, abstractmethod
import math
import torch
import torch.nn as nn
from .multimodal_encoder.builder import build_vision_tower
from ChatUniVi.constants import *
from .cluster import CTM, TCBlock, TokenMergeClusterDPCKNN, UnitTokenMergeClusterDPCKNN, VideoTokenMergeClusterDPCKNN, create_token_dict_from_features
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
        if not getattr(self, method_name):
            raise ValueError(f"[MetaModel.initialize_cluster_modules] {method_name} not supported.")

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
        )
        self.token_merging_video = TokenMergeClusterDPCKNN(
            sample_ratios=sample_ratios_video,
            ks=ks[:len(sample_ratios_video)],
            coord_weight=coord_weight_video,
            token_ordering=token_ordering_video,
        )




class ChatUniViMetaForCausalLM(ABC):
    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def encode_images(self, images):
        image_features = self.get_model().get_vision_tower()(images, select_feature="patch")
        return image_features

    def positional_encoding(self, x, num_features=1024, max_len=64):
        p = torch.zeros((1, max_len, num_features))
        _x = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1) / torch.pow(10000,
                                                                            torch.arange(0, num_features, 2, dtype=torch.float32) / num_features)

        p[:, :, 0::2] = torch.sin(_x)
        p[:, :, 1::2] = torch.cos(_x)
        x = x + p[:, :x.shape[1], :].to(x.device).to(x.dtype)
        return x


    def project(self, image_features, input_type="image"):
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
            image_features = self.get_model().mm_projector(image_features)
            return image_features
        
        method_name = f"project_{self.get_model().cluster_type}"
        if not getattr(self, method_name):
            raise ValueError(f"[ChatUniViMetaForCausalLM.project] {method_name} not supported.")
    
        image_features = getattr(self, method_name)(image_features, input_type=input_type)
        image_features = image_features.to(self.get_model().mm_projector.weight.dtype)
        image_features = self.get_model().mm_projector(image_features)
        return image_features


    def project_v3(self, image_features, input_type="image"):
        if input_type == "image":
            sizes = (int(math.sqrt(image_features.shape[1])),) * 2
            token_dict = create_token_dict_from_features(image_features, sizes)
            token_dict_list = self.get_model().token_merge_image(token_dict)[1:]
            image_features = torch.cat([d['x'] for d in token_dict_list], dim=1)
        else:
            sizes = (image_features.shape[0],) + (int(image_features.shape[1] ** (1 / 2)),)*2
            token_dict = create_token_dict_from_features(image_features.reshape(1, -1, image_features.shape[-1]), sizes)
            token_dict_list = self.get_model().token_merging_video(token_dict)[1:]
            image_features = torch.cat([d['x'] for d in token_dict_list], dim=1)
        return image_features


    def project_v2(self, image_features, input_type="image"):
        if input_type == "image":
            sizes = (int(math.sqrt(image_features.shape[1])),) * 2
            token_dict = create_token_dict_from_features(image_features, sizes)
            token_dict_image_list = self.get_model().token_merging_model.token_merge_image(token_dict)[1:]
            # multiscale features: (B, 112, C)
            image_features = torch.cat([d['x'] for d in token_dict_image_list], dim=1)
        else:
            outputs = self.get_model().token_merging_model(image_features)
            image_features = torch.cat([torch.cat([d['x'] for d in l], dim=1) for l in outputs['token_dict_video_list']], dim=1)
        return image_features


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
        self, input_ids, attention_mask, past_key_values, labels, images
    ):
        # images: 
        
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            if past_key_values is not None and vision_tower is not None and images is not None and input_ids.shape[1] == 1:
                attention_mask = torch.ones((attention_mask.shape[0], past_key_values[-1][-1].shape[-2] + 1), dtype=attention_mask.dtype, device=attention_mask.device)
            return input_ids, attention_mask, past_key_values, None, labels

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

            # image_token_indices: tensor([35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48], device='cuda:0')
            image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]

            cur_new_input_embeds = []
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


                for i in temp:
                    image_token_start = image_token_indices[0] # tensor(35, device='cuda:0')
                    image_token_end = image_token_indices[-1]  # tensor(57, device='cuda:0')
                    cur_image_features = []

                    for _ in i:
                        cur_image_features.append(image_features[cur_image_idx])
                        cur_image_idx += 1

                    # cur_image_features: [(576, 1024), ..., (576, 1024)]. features for each frame in the video.

                    if len(i) > 2:
                        cur_image_features = torch.stack(cur_image_features, dim=0)

                    # cur_image_features: (#frames, 576, 1024)

                        cur_image_features = self.project(cur_image_features, input_type="video")
                        t, l, n = cur_image_features.size()
                        cur_image_features = cur_image_features.contiguous().view(t * l, n)
                    else:
                        cur_image_features = torch.stack(cur_image_features, dim=0)
                        cur_image_features = self.project(cur_image_features, input_type="image")
                        t, l, n = cur_image_features.size()
                        cur_image_features = cur_image_features.contiguous().view(t * l, n)

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
                    else:
                        

                        # import json, logging
                        # logging.info('prepare_inputs_labels_for_multimodal before apply embed_tokens to cur_input_ids: '+ json.dumps({
                        #     'self.device': str(self.device),
                        #     'mm_projector.device': str(self.get_model().mm_projector.weight.device),
                        #     'embed_tokens.device': str(self.get_model().embed_tokens.weight.device),
                        #     'input_ids.device': str(input_ids.device),
                        #     'images.device': str(images.device),
                        #     'attention_mask.device': str(attention_mask.device),
                        #     'image_features.device': str(image_features.device),
                        #     'cur_input_ids': str(cur_input_ids[:image_token_start].device),
                        # }, indent=4))

                        cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[:image_token_start]))
                        cur_new_input_embeds.append(cur_image_features)
                        if labels is not None:
                            cur_new_labels.append(cur_labels[:image_token_start])
                            cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=labels.device, dtype=labels.dtype))
                            cur_labels = cur_labels[image_token_end + 1:]

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
                cur_image_features = self.project(cur_image_features, input_type="image")
                t, l, n = cur_image_features.size()
                # (B*112, D)
                cur_image_features = cur_image_features.contiguous().view(t * l, n)

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
                else:
                    # this branch since `tune_mm_mlp_adapter` typically set to False
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[:image_token_start]))
                    cur_new_input_embeds.append(cur_image_features)
                    if labels is not None:
                        cur_new_labels.append(cur_labels[:image_token_start])
                        cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=labels.device, dtype=labels.dtype))
                        cur_labels = cur_labels[image_token_end+1:]

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
            cur_new_input_embeds = [x.to(device=self.device) for x in cur_new_input_embeds]
            cur_new_input_embeds = torch.cat(cur_new_input_embeds, dim=0)
            new_input_embeds.append(cur_new_input_embeds)
            if labels is not None:
                cur_new_labels = torch.cat(cur_new_labels, dim=0)
                new_labels.append(cur_new_labels)


        # does padding to longest sequence.
        if any(x.shape != new_input_embeds[0].shape for x in new_input_embeds):
            max_len = max(x.shape[0] for x in new_input_embeds)

            new_input_embeds_align = []
            for cur_new_embed in new_input_embeds:
                cur_new_embed = torch.cat((cur_new_embed, torch.zeros((max_len - cur_new_embed.shape[0], cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)), dim=0)
                new_input_embeds_align.append(cur_new_embed)
            new_input_embeds = torch.stack(new_input_embeds_align, dim=0)

            if labels is not None:
                new_labels_align = []
                _new_labels = new_labels
                for cur_new_label in new_labels:
                    cur_new_label = torch.cat((cur_new_label, torch.full((max_len - cur_new_label.shape[0],), IGNORE_INDEX, dtype=cur_new_label.dtype, device=cur_new_label.device)), dim=0)
                    new_labels_align.append(cur_new_label)
                new_labels = torch.stack(new_labels_align, dim=0)

            if attention_mask is not None:
                new_attention_mask = []
                for cur_attention_mask, cur_new_labels, cur_new_labels_align in zip(attention_mask, _new_labels, new_labels):
                    new_attn_mask_pad_left = torch.full((cur_new_labels.shape[0] - labels.shape[1],), True, dtype=attention_mask.dtype, device=attention_mask.device)
                    new_attn_mask_pad_right = torch.full((cur_new_labels_align.shape[0] - cur_new_labels.shape[0],), False, dtype=attention_mask.dtype, device=attention_mask.device)
                    cur_new_attention_mask = torch.cat((new_attn_mask_pad_left, cur_attention_mask, new_attn_mask_pad_right), dim=0)
                    new_attention_mask.append(cur_new_attention_mask)
                attention_mask = torch.stack(new_attention_mask, dim=0)
                assert attention_mask.shape == new_labels.shape
        else:
            new_input_embeds = torch.stack(new_input_embeds, dim=0)
            if labels is not None:
                new_labels = torch.stack(new_labels, dim=0)

            if attention_mask is not None:
                new_attn_mask_pad_left = torch.full((attention_mask.shape[0], new_input_embeds.shape[1] - input_ids.shape[1]), True, dtype=attention_mask.dtype, device=attention_mask.device)
                attention_mask = torch.cat((new_attn_mask_pad_left, attention_mask), dim=1)
                assert attention_mask.shape == new_input_embeds.shape[:2]

        return None, attention_mask, past_key_values, new_input_embeds, new_labels

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
                    p.requires_grad = False
                    
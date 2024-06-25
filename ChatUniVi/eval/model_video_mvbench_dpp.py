# wpq: taken from https://github.com/mbzuai-oryx/VideoGPT-plus/blob/main/eval/mvbench/inference/infer.py
#
import argparse
import torch
import os
import json
import subprocess
from tqdm import tqdm
import shortuuid
from ChatUniVi.constants import *
from ChatUniVi.conversation import conv_templates, SeparatorStyle
from ChatUniVi.model.builder import load_pretrained_model
from ChatUniVi.utils import disable_torch_init
from ChatUniVi.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from ChatUniVi.eval.video_encoding import _get_rawvideo_dec, read_frame_mod, read_gif_mod
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import traceback



def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)



def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print



def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
        args.dist_url = 'env://'
        os.environ['LOCAL_SIZE'] = str(torch.cuda.device_count())
        print('Using distributed mode: 1')
    elif 'SLURM_PROCID' in os.environ:
        num_gpus = torch.cuda.device_count()
        proc_id = int(os.environ['SLURM_PROCID'])
        ntasks = int(os.environ['SLURM_NTASKS'])
        node_list = os.environ['SLURM_NODELIST']
        num_gpus = torch.cuda.device_count()
        addr = subprocess.getoutput(
            'scontrol show hostname {} | head -n1'.format(node_list))
        os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '3461')
        os.environ['MASTER_ADDR'] = addr
        os.environ['WORLD_SIZE'] = str(ntasks)
        os.environ['RANK'] = str(proc_id)
        os.environ['LOCAL_RANK'] = str(proc_id % num_gpus)
        os.environ['LOCAL_SIZE'] = str(num_gpus)
        args.dist_url = 'env://'
        args.world_size = ntasks
        args.rank = proc_id
        args.gpu = proc_id % num_gpus
        print('Using distributed mode: slurm')
        print(f"world: {os.environ['WORLD_SIZE']}, rank:{os.environ['RANK']},"
              f" local_rank: {os.environ['LOCAL_RANK']}, local_size: {os.environ['LOCAL_SIZE']}")
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    # setup_for_distributed(args.rank == 0)
    # setup_for_distributed(args.rank == 1)

    import logging
    logging.basicConfig(level=logging.INFO, filemode='w', filename=f'process_{args.gpu}.log')



def qa_template(data):
    question = f"Question: {data['question']}\n"
    question += "Options:\n"
    answer = data['answer']
    answer_idx = -1
    for idx, c in enumerate(data['candidates']):
        question += f"({chr(ord('A') + idx)}) {c}\n"
        if c == answer:
            answer_idx = idx
    question = question.rstrip()
    answer = f"({chr(ord('A') + answer_idx)}) {answer}"

    # Add the instruction to question
    question_prompt = "\nOnly give the best option."  # to change
    question += question_prompt

    return question, answer



class EvalDatasetMvBench(Dataset):
    def __init__(self, gt_dir, video_dir, image_processor, mvbench_data_list):
        self.gt_contents = []
        for k, v in mvbench_data_list.items():
            with open(os.path.join(gt_dir, v[0]), 'r') as f:
                json_data = json.load(f)
            for data in json_data:
                self.gt_contents.append(
                    {'task_type': k, 'prefix': v[1], 'data_type': v[2], 'bound': v[3], 'data': data}
                )
        self.video_dir = video_dir
        self.image_processor = image_processor

    def __len__(self):
        return len(self.gt_contents)

    def __getitem__(self, idx):
        sample = self.gt_contents[idx]

        task_type = sample['task_type']

        if sample['bound']:
            bound = (sample['data']['start'], sample['data']['end'],)
        else:
            bound = (None, None)
        data_type = sample['data_type']
        prefix = sample['prefix']
        video_name = sample['data']['video']
        video_path = os.path.join(self.video_dir, prefix, video_name)
        if os.path.exists(video_path):
            if data_type == 'video':
                video_frames, slice_len = _get_rawvideo_dec(video_path, self.image_processor, s=bound[0], e=bound[1], max_frames=MAX_IMAGE_LENGTH, num_threads=(1 if task_type=='Action Antonym' else 0))
            elif data_type == 'gif':
                video_frames, slice_len = read_gif_mod(video_path, self.image_processor, s=bound[0], e=bound[1], max_frames=MAX_IMAGE_LENGTH)
            elif data_type == 'frame':
                video_frames, slice_len = read_frame_mod(video_path, self.image_processor, s=bound[0], e=bound[1], max_frames=MAX_IMAGE_LENGTH, image_resolution=224)
        else:
            video_frames, slice_len = "None", 0
            print('Video not found:', video_path)

        sample_set = {}
        question, answer = qa_template(sample['data'])
        sample_set['video_name'] = f'{prefix}_{video_name}'
        sample_set['Q'] = question
        sample_set['A'] = answer
        sample_set['task_type'] = task_type

        return idx, [sample_set], video_frames, slice_len
    



mvbench_data_list = {
    "Episodic Reasoning": ("episodic_reasoning.json", "tvqa/frames_fps3_hq/", "frame", True),
    "Action Sequence": ("action_sequence.json", "star/Charades_v1_480/", "video", True),
    "Action Prediction": ("action_prediction.json", "star/Charades_v1_480/", "video", True),
    "Action Antonym": ("action_antonym.json", "ssv2_video/", "video", False),
    "Fine-grained Action": ("fine_grained_action.json", "Moments_in_Time_Raw/videos/", "video", False),
    "Unexpected Action": ("unexpected_action.json", "FunQA_test/test/", "video", False),
    "Object Existence": ("object_existence.json", "clevrer/video_validation/", "video", False),
    "Object Interaction": ("object_interaction.json", "star/Charades_v1_480/", "video", True),
    "Object Shuffle": ("object_shuffle.json", "perception/videos/", "video", False),
    "Moving Direction": ("moving_direction.json", "clevrer/video_validation/", "video", False),
    "Action Localization": ("action_localization.json", "sta/sta_video/", "video", True),
    "Scene Transition": ("scene_transition.json", "scene_qa/video/", "video", False),
    "Action Count": ("action_count.json", "perception/videos/", "video", False),
    "Moving Count": ("moving_count.json", "clevrer/video_validation/", "video", False),
    "Moving Attribute": ("moving_attribute.json", "clevrer/video_validation/", "video", False),
    "State Change": ("state_change.json", "perception/videos/", "video", False),
    "Fine-grained Pose": ("fine_grained_pose.json", "nturgbd/", "video", False),
    "Character Order": ("character_order.json", "perception/videos/", "video", False),
    "Egocentric Navigation": ("egocentric_navigation.json", "vlnqa/", "video", False),
    "Counterfactual Inference": ("counterfactual_inference.json", "clevrer/video_validation/", "video", False),
}



def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
    mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
    if mm_use_im_patch_token:
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    if mm_use_im_start_end:
        tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
    model.resize_token_embeddings(len(tokenizer))

    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model()
    image_processor = vision_tower.image_processor

    if model.config.config["use_cluster"]:
        for n, m in model.named_modules():
            m = m.to(dtype=torch.bfloat16)
    
    device = f'cuda:{args.gpu}' if args.world_size > 1 else 'cuda'
    model = model.to(device)


    dataset = EvalDatasetMvBench(args.question_dir, args.video_folder, image_processor, mvbench_data_list)
    distributed_sampler = DistributedSampler(dataset, rank=args.rank, num_replicas=args.world_size, shuffle=False)
    dataloader = DataLoader(dataset, batch_size=args.batch_size_per_gpu, num_workers=8, sampler=distributed_sampler)

    for (idx, sample_set, video_frames, slice_len) in tqdm(dataloader):
        idx, sample_set, video_frames, slice_len = int(idx[0]), sample_set[
            0], video_frames, int(slice_len[0])

        sample = sample_set
        qs = sample['Q'][0]

        # try:
        cur_prompt = qs
        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN * slice_len + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN * slice_len + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX,
                                            return_tensors='pt').unsqueeze(0).to(device)

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        # wpq: since `_get_rawvideo_dec` is modified, need to concat images before feeding into `.generate`
        images = torch.cat(video_frames, dim=0).half().to(device)

        with torch.inference_mode():

            # import json, logging
            # logging.info('before model.generate: '+ json.dumps({
            #     'self.device': str(model.device),
            #     'mm_projector.device': str(model.get_model().mm_projector.weight.device),
            #     'embed_tokens.device': str(model.get_model().embed_tokens.weight.device),
            #     'input_ids.device': str(input_ids.device),
            #     'images.device': str(images.device),
            # }, indent=4))

            output_ids = model.generate(
                input_ids,
                images=images,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=1024,
                use_cache=True,
                stopping_criteria=[stopping_criteria])

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()
        # wpq: not necessary for now. this prob used for phi-3
        # outputs = outputs.replace("<|end|>", '')
        # outputs = outputs.strip()

        ans_id = shortuuid.uuid()
        video_json_name = sample['video_name'][0].replace('/', '_')
        # wpq: ensure every file is unique
        # if len(video_json_name) > 100:
        #     video_json_name = video_json_name[50:]

        results = {'video_name': sample['video_name'][0],
                    "prompt": cur_prompt,
                    "pred": outputs,
                    "answer_id": ans_id,
                    "Q": sample_set['Q'][0],
                    "task_type": sample['task_type'][0],
                    "A": sample['A'][0]}
        with open(f"{args.output_dir}/answers/{video_json_name}_{idx}.json", "w") as f:
            json.dump(results, f)

        # except Exception as e:
        #     trace = traceback.format_exc()
        #     print(f"Error processing video file '{sample['video_name'][0]}': {e}")
        #     print("Detailed traceback:")
        #     print(trace)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="MBZUAI/VideoGPT-plus_Phi3-mini-4k/mvbench")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--video-folder", type=str, default="OpenGVLab/MVBench/video")
    parser.add_argument("--question-dir", type=str, default="OpenGVLab/MVBench/json")
    parser.add_argument("--output-dir", type=str, default="MBZUAI/VideoGPT-plus_Phi3-mini-4k/mvbench_eval")
    parser.add_argument("--conv-mode", type=str, default="phi3_instruct")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)

    parser.add_argument("--batch_size_per_gpu", type=int, required=False, default=1)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_url', type=str, default='env://', help='url used to set up distributed training')

    args = parser.parse_args()

    init_distributed_mode(args)

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'answers'), exist_ok=True)

    eval_model(args)

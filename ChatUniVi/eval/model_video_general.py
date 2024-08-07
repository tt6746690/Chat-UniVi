import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
from ChatUniVi.constants import *
from ChatUniVi.conversation import conv_templates, SeparatorStyle
from ChatUniVi.model.builder import load_pretrained_model
from ChatUniVi.utils import disable_torch_init
from ChatUniVi.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from ChatUniVi.model.dataloader import _get_rawvideo_dec
from PIL import Image
import math
from decord import VideoReader, cpu
import numpy as np
import random

def get_chunk(lst, n, k, seed=0):
    """randomize the examples ordering. """
    random.seed(seed)
    indices = list(range(len(lst)))
    random.shuffle(indices)  # Shuffle the indices deterministically
    chunk_size = math.ceil(len(lst) / n) # integer division
    chunks = [indices[i:i + chunk_size] for i in range(0, len(indices), chunk_size)]
    chunk = [lst[idx] for idx in chunks[k]]
    return chunk


def eval_model(args):

    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = "ChatUniVi"
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    # Load the ground truth file
    with open(args.question_file) as file:
        gt_contents = json.load(file)
    gt_contents = get_chunk(gt_contents, args.num_chunks, args.chunk_idx)

    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    video_formats = ['.mp4', '.avi', '.mov', '.mkv']

    # Iterate over each sample in the ground truth file
    for sample in tqdm(gt_contents):
        video_name = sample['video_name']
        sample_set = sample
        qs = sample['Q']

        # Load the video file
        for fmt in video_formats:  # Added this line
            temp_path = os.path.join(args.video_folder, f"{video_name}{fmt}")
            if os.path.exists(temp_path):
                video_path = temp_path
                break

        # Check if the video exists
        if video_path is not None:  # Modified this line
            video_frames, slice_len = _get_rawvideo_dec(video_path, image_processor, max_frames=MAX_IMAGE_LENGTH)

        try:
            cur_prompt = qs
            if model.config.mm_use_im_start_end:
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN * slice_len + DEFAULT_IM_END_TOKEN + '\n' + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN * slice_len + '\n' + qs

            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(
                0).cuda()

            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

            # wpq: since `_get_rawvideo_dec` is modified, need to concat images before feeding into `.generate`
            images = torch.stack(video_frames).half().to('cuda')
            print(images.shape, images.device)
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=images,
                    do_sample=True if args.temperature!=0. else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    max_new_tokens=1024,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria],
                    matryoshka_vis_token_scale = getattr(args, "matryoshka_vis_token_scale", None),
                    )


            input_token_len = input_ids.shape[1]
            n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
            if n_diff_input_output > 0:
                print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
            outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
            outputs = outputs.strip()
            if outputs.endswith(stop_str):
                outputs = outputs[:-len(stop_str)]
            outputs = outputs.strip()

            ans_id = shortuuid.uuid()
            ans_file.write(json.dumps({'video_name': sample['video_name'],
                                       "prompt": cur_prompt,
                                       "text": outputs,
                                       "answer_id": ans_id,
                                       "model_id": model_name,
                                       "answer": sample['A'],
                                       "metadata": {}}) + "\n")
            ans_file.flush()
        except Exception as e:
            print(f"Error processing video file '{video_name}': {e}")

    ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--video-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--model_use", type=str, default="BASE")
    parser.add_argument("--matryoshka_vis_token_scale", type=str, default=None)
    args = parser.parse_args()

    eval_model(args)

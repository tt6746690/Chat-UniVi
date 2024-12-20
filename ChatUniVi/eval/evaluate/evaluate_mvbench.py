import os
import json
from tqdm import tqdm
import argparse


def check_ans(pred, gt):
    flag = False

    pred_list = pred.lower().split(' ')
    pred_option = pred_list[0]
    # (a) table is blue -> ['(a)', 'table', 'is', 'blue']
    gt_list = gt.lower().split(' ') 
    # '(a)'
    gt_option = gt_list[0] 
    # if gt_content[-1] == '.':
    #     gt_content = gt_content[:-1]
    # print('check_ans: ', gt_option, pred_option)
    if pred_option.replace('.', '') in gt_option:
        flag = True
    elif gt_option in pred_option:
        flag = True

    return flag


def main(args):
    result_files = os.listdir(os.path.join(args.output_dir, 'answers'))

    correct = 0
    total = 0
    acc_dict = {}
    data_dict = []

    for file in tqdm(result_files):
        if file.endswith('.json'):
            json_file = os.path.join(args.output_dir, 'answers', file)
            json_data = json.load(open(json_file))
            video_name = json_data['video_name']
            task_type = json_data['task_type']
            pred = json_data['pred']
            gt_answer = json_data['A']
            question = json_data['Q']

            if task_type not in acc_dict:
                acc_dict[task_type] = [0, 0]  # correct, total
            acc_dict[task_type][1] += 1
            total += 1

            ans = check_ans(pred=pred, gt=gt_answer)
            if ans:
                acc_dict[task_type][0] += 1
                correct += 1

            json_data.update({'ans': ans})
            data_dict.append(json_data)

    types = {'Action Sequence': 0, 'Action Prediction': 0, 'Action Antonym': 0, 'Fine-grained Action': 0,
             'Unexpected Action': 0, 'Object Existence': 0, 'Object Interaction': 0, 'Object Shuffle': 0,
             'Moving Direction': 0, 'Action Localization': 0, 'Scene Transition': 0, 'Action Count': 0,
             'Moving Count': 0, 'Moving Attribute': 0, 'State Change': 0, 'Fine-grained Pose': 0, 'Character Order': 0,
             'Egocentric Navigation': 0, 'Episodic Reasoning': 0, 'Counterfactual Inference': 0}

    result_list = []
    result_dict = {}
    for task_type, v in types.items():
        print('-' * 30, task_type, '-' * 30)
        Acc = acc_dict[task_type][0] / acc_dict[task_type][1] * 100
        print(f"{task_type}  Acc: {Acc :.2f}%")
        result_list.append(Acc)
        result_dict[task_type] = Acc
    print(f"All Acc: {result_list}%")
    print(f"Total Acc: {correct / total * 100 :.2f}%")
    result_dict['total'] = correct / total * 100

    with open(os.path.join(args.output_dir, 'metrics.json'), 'w') as f:
        json.dump(result_dict, f)

    with open(os.path.join(args.output_dir, 'answers.json'), 'w') as f:
        json.dump(data_dict, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="MBZUAI/VideoGPT-plus_Phi3-mini-4k/mvbench_eval")
    args = parser.parse_args()

    main(args)
# wpq: modify data path
PATH = '/fsx/wpq/.data/chatunivi/'


Pretrain = {
    "chat_path": f"{PATH}/CC3M-595K/chat.json",
    "CC3M": f"{PATH}/CC3M-595K",
}

VIT = {
    "chat_path": f"{PATH}/llava_instruct_150k.json",
    "COCO2017": f"{PATH}/COCO2017/train2017",
}

MIMIC_imageonly = {
    "chat_path": f"{PATH}/MIMIC-IT-imageonly.json",
    "CDG": f"{PATH}/CGD/images",
    "LA": f"{PATH}/LA/images",
    "SD": f"{PATH}/SD/images",
}

COCO_CAP = {
    "chat_path": f"{PATH}/COCO/coco_cap_chat.json",
    "COCO2014": f"{PATH}/COCO2014/train2014",
}

COCO_REG = {
    "chat_path": f"{PATH}/COCO/coco_reg_chat.json",
    "COCO2014": f"{PATH}/COCO2014/train2014",
}

COCO_REC = {
    "chat_path": f"{PATH}/COCO/coco_rec_chat.json",
    "COCO2014": f"{PATH}/COCO2014/train2014",
}

VIDEO = {
    "chat_path": f"{PATH}/video_chat.json",
    "VIDEO": f"{PATH}/Activity_Videos",
}

SQA = {
    "chat_path": f"{PATH}/llava_train_QCM-LEA.json",
    "ScienceQA": f"{PATH}/scienceqa/train",
}

Pretrain_valley_llava = {
    "chat_path": f"{PATH}/valley_llavaimage.json",
    "valley": f"{PATH}/Data",
    "llava": f"{PATH}/Data",  # from llava v1.5
}

LLaVA = {
    "chat_path": f"{PATH}/llavaimage_tune.json",
    "llava": f"{PATH}/Data",  # from llava v1.5
}


# "Pretrainv1.5": [Pretrain, Pretrain_valley_llava],
# "FINETUNEv1.5": [VIT, VIDEO, LLaVA],

# wpq: just add llava v1.5 dataset
# each dictionary contains `chat_path` that points to the questions
# the rest kv pairs are image folders.


pretrain_valley = {
    "chat_path": "/fsx/wpq/.data/chatunivi/Chat-UniVi-Instruct/v1.5_train_json/valley_existfiltered.json",
    "valley": "/fsx/wpq/.data/videollava/",
}

pretrain_valley_300k = {
    "chat_path": "/fsx/wpq/.data/chatunivi/Chat-UniVi-Instruct/v1.5_train_json/valley_existfiltered_300k.json",
    "valley": "/fsx/wpq/.data/videollava/",
}

pretrain_llava = {
    "chat_path": "/fsx/wpq/.data/chatunivi/Chat-UniVi-Instruct/v1.5_train_json/llavaimage_existfiltered.json",
    "llava": "/fsx/wpq/.data/videollava/",
}

pretrain_llava_300k = {
    "chat_path": "/fsx/wpq/.data/chatunivi/Chat-UniVi-Instruct/v1.5_train_json/llavaimage_existfiltered_300k.json",
    "llava": "/fsx/wpq/.data/videollava/",
}

Pretrain_valley_llava = {
    # filter the json to remove ones where valley videos are missing.
    "chat_path": "/fsx/wpq/.data/chatunivi/Chat-UniVi-Instruct/v1.5_train_json/valley_llavaimage_existfiltered.json",
    # following two get image/video from Video-LLaVA hf dataset repo.
    # should contain folder: `valley` 
    "valley": "/fsx/wpq/.data/videollava/",
    # should contain folder: `llava_image`
    "llava": "/fsx/wpq/.data/videollava/",
}

Pretrain_valley_llava_300k = {
    "chat_path": "/fsx/wpq/.data/chatunivi/Chat-UniVi-Instruct/v1.5_train_json/valley_llavaimage_existfiltered_300k.json",
    "valley": "/fsx/wpq/.data/videollava/",
    "llava": "/fsx/wpq/.data/videollava/",
}

LLaVA = {
    "chat_path": "/fsx/wpq/.data/chatunivi/Chat-UniVi-Instruct/v1.5_train_json/llavaimage_tune.json",
    # following two get image/video from Video-LLaVA hf dataset repo.
    # should contain folder: `llava_image_tune`
    "llava": "/fsx/wpq/.data/videollava/",
}

LLaVA200k = {
    "chat_path": "/fsx/wpq/.data/chatunivi/Chat-UniVi-Instruct/v1.5_train_json/llavaimage_tune_200k.json",
    "llava": "/fsx/wpq/.data/videollava/",
}


VIDEO = {
    "chat_path": "/fsx/wpq/.data/chatunivi/Chat-UniVi-Instruct/Fine-tuning/VIDEO/video_chat.json",
    "VIDEO": "/fsx/wpq/.data/chatunivi/Chat-UniVi-Instruct/Fine-tuning/VIDEO/Activity_Videos",
}

 
VIT = {
    "chat_path": "/fsx/wpq/.data/chatunivi/Chat-UniVi-Instruct/Fine-tuning/VIT/llava_instruct_150k.json",
    "COCO2017": "/datasets01/COCO/022719/train2017",
}


# MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
# # torchrun --nproc_per_node=1 ChatUniVi/eval/model_video_mvbench.py \
# torchrun --nnodes=1 --nproc_per_node=$CHUNKS --rdzv_backend=c10d --rdzv-endpoint=$MASTER_ADDR:$MASTER_PORT
    # torchrun --nnodes=1 --nproc_per_node=2 --rdzv_backend=c10d --rdzv-endpoint=$MASTER_ADDR:1234 \
# CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 \
#     ChatUniVi/eval/model_video_mvbench.py \
#     --model-path $CKPT \
#     --video-folder /fsx/wpq/.data/chatunivi/eval/MVBench/video \
#     --question-dir /fsx/wpq/.data/chatunivi/eval/MVBench/json_valid \
#     --output-dir $CKPT/eval/mvbench \
#     --temperature 0.0 \
#     --conv-mode v1 \
#     --batch_size_per_gpu 1

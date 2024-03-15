python train_pointwise.py \
    --model  "/data1/linzelun/workspace/pretrained_model_torch/sbert-base-chinese-nli" \
    --train_path "data/comment_classify3/train.txt" \
    --dev_path "data/comment_classify3/dev.txt" \
    --save_dir "checkpoints/comment_classify3" \
    --img_log_dir "logs/comment_classify3" \
    --img_log_name "ERNIE-PointWise" \
    --batch_size 8 \
    --max_seq_len 128 \
    --valid_steps 50 \
    --logging_steps 10 \
    --num_train_epochs 5 \
    --device "cuda:5"


    # "/data1/linzelun/workspace/pretrained_model_torch/sbert-base-chinese-nli"


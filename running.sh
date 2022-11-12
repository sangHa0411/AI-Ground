# Training with Evaluation
python train.py --data_dir ./data \
    --meta_data_file meta_data.csv \
    --profile_data_file profile_data.csv \
    --history_data_file history_data.csv \
    --max_length 60 \
    --do_eval True \
    --learning_rate 2e-4 \
    --weight_decay 1e-3 \
    --train_batch_size 128 \
    --eval_batch_size 32 \
    --epochs 50 \
    --mlm_probability 0.4 \
    --num_workers 6 \
    --logging_steps 100 \
    --eval_steps 500 \
    --hidden_size 512 \
    --num_layers 3 \
    --num_head 4 \
    --intermediate_size 2048 \
    --dropout_prob 0.1 \
    --warmup_ratio 0.1 \
    --eps 1e-8 \
    --save_dir ./exps
    

# Full Training
rm -rf ./exps/*
python train.py --data_dir ./data \
    --meta_data_file meta_data.csv \
    --profile_data_file profile_data.csv \
    --history_data_file history_data.csv \
    --max_length 200 \
    --learning_rate 1e-4 \
    --weight_decay 1e-3 \
    --train_batch_size 128 \
    --eval_batch_size 32 \
    --epochs 50 \
    --save_steps 500 \
    --mlm_probability 0.2 \
    --num_workers 6 \
    --logging_steps 100 \
    --eval_steps 500 \
    --hidden_size 256 \
    --num_layers 6 \
    --num_head 8 \
    --intermediate_size 1024 \
    --dropout_prob 0.1 \
    --warmup_ratio 0.1 \
    --eps 1e-8 \
    --save_dir ./exps


# Prediction
python predict.py  --data_dir ./data \
    --meta_data_file meta_data.csv \
    --submission_file sample_submission.csv \
    --profile_data_file profile_data.csv \
    --history_data_file history_data.csv \
    --max_length 200 \
    --eval_batch_size 32 \
    --model_path exps/checkpoint-1950.pt \
    --num_workers 6 \
    --hidden_size 512 \
    --num_layers 3 \
    --num_head 4 \
    --intermediate_size 2048 \
    --dropout_prob 0.1 \
    --warmup_ratio 0.1 \
    --eps 1e-8 \
    --output_file ./results/2022_11_12_3.csv
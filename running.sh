# Training with Evaluation

## Model 1
python train.py --data_dir ./data \
    --meta_data_file meta_data.csv \
    --profile_data_file profile_data.csv \
    --history_data_file history_data.csv \
    --max_length 50 \
    --do_eval True \
    --learning_rate 1e-4 \
    --weight_decay 1e-3 \
    --train_batch_size 128 \
    --eval_batch_size 32 \
    --epochs 50 \
    --mlm_probability 0.4 \
    --logging_steps 100 \
    --eval_steps 500 \
    --hidden_size 768 \
    --num_layers 12 \
    --num_head 8 \
    --num_workers 3 \
    --intermediate_size 3072 \
    --dropout_prob 0.1 \
    --warmup_ratio 0.1 \
    --eps 1e-12 \
    --save_dir ./exps

## Model 2
python train.py --data_dir ./data \
    --meta_data_file meta_data.csv \
    --profile_data_file profile_data.csv \
    --history_data_file history_data.csv \
    --max_length 50 \
    --do_eval True \
    --learning_rate 1.2e-4 \
    --weight_decay 1e-2 \
    --train_batch_size 128 \
    --eval_batch_size 32 \
    --max_steps 2600 \
    --mlm_probability 0.4 \
    --logging_steps 100 \
    --eval_steps 500 \
    --hidden_size 768 \
    --num_layers 12 \
    --num_head 8 \
    --num_workers 3 \
    --intermediate_size 3072 \
    --dropout_prob 0.1 \
    --warmup_ratio 0.1 \
    --eps 1e-12 \
    --save_dir ./exps
    

## Model 3
python train.py --data_dir ./data \
    --meta_data_file meta_data.csv \
    --profile_data_file profile_data.csv \
    --history_data_file history_data.csv \
    --max_length 50 \
    --do_eval True \
    --learning_rate 5e-5 \
    --weight_decay 1e-2 \
    --train_batch_size 128 \
    --eval_batch_size 32 \
    --epochs 50 \
    --max_steps 4000 \
    --mlm_probability 0.4 \
    --logging_steps 100 \
    --eval_steps 500 \
    --hidden_size 768 \
    --num_layers 12 \
    --num_head 8 \
    --num_workers 3 \
    --intermediate_size 3072 \
    --dropout_prob 0.1 \
    --warmup_ratio 0.1 \
    --eps 1e-12 \
    --save_dir ./exps


# Full Training
rm -rf ./exps/*
python train.py --data_dir ./data \
    --meta_data_file meta_data.csv \
    --profile_data_file profile_data.csv \
    --history_data_file history_data.csv \
    --max_length 50 \
    --learning_rate 1e-4 \
    --weight_decay 1e-3 \
    --train_batch_size 128 \
    --epochs 50 \
    --mlm_probability 0.4 \
    --save_steps 500 \
    --logging_steps 100 \
    --hidden_size 768 \
    --num_layers 12 \
    --num_head 8 \
    --num_workers 3 \
    --intermediate_size 3072 \
    --dropout_prob 0.1 \
    --warmup_ratio 0.1 \
    --eps 1e-12 \
    --save_dir ./exps


# Prediction
python predict.py  --data_dir ./data \
    --meta_data_file meta_data.csv \
    --submission_file sample_submission.csv \
    --profile_data_file profile_data.csv \
    --history_data_file history_data.csv \
    --max_length 50 \
    --eval_batch_size 32 \
    --model_path exps/checkpoint-3250.pt \
    --num_workers 3 \
    --hidden_size 768 \
    --num_layers 12 \
    --num_head 8 \
    --intermediate_size 3072 \
    --dropout_prob 0.1 \
    --eps 1e-12 \
    --output_file ./results/2022_11_15_0.csv
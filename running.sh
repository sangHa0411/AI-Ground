# Training with Evaluation

## Model 1
python train.py --data_dir ./data \
    --seed 42 \
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
    --max_steps 3000 \
    --mlm_probability 0.4 \
    --logging_steps 100 \
    --eval_steps 500 \
    --hidden_size 768 \
    --num_layers 12 \
    --num_head 8 \
    --num_workers 4 \
    --intermediate_size 3072 \
    --dropout_prob 0.1 \
    --warmup_ratio 0.1 \
    --eps 1e-12 \
    --save_dir ./exps

# Full Training
rm -rf ./exps/*

## Model 1
mkdir ./exps/seed42
python train.py --data_dir ./data \
    --meta_data_file meta_data.csv \
    --profile_data_file profile_data.csv \
    --history_data_file history_data.csv \
    --seed 42 \
    --max_length 50 \
    --learning_rate 1e-4 \
    --weight_decay 1e-3 \
    --train_batch_size 128 \
    --eval_batch_size 32 \
    --epochs 50 \
    --max_steps 3000 \
    --mlm_probability 0.4 \
    --logging_steps 100 \
    --save_steps 500 \
    --hidden_size 768 \
    --num_layers 12 \
    --num_head 8 \
    --num_workers 3 \
    --intermediate_size 3072 \
    --dropout_prob 0.1 \
    --warmup_ratio 0.1 \
    --eps 1e-12 \
    --save_dir ./exps/seed42

## Model 2
mkdir ./exps/seed1234
python train.py --data_dir ./data \
    --meta_data_file meta_data.csv \
    --profile_data_file profile_data.csv \
    --history_data_file history_data.csv \
    --seed 1234 \
    --epochs 50 \
    --max_length 50 \
    --learning_rate 1e-4 \
    --weight_decay 1e-3 \
    --train_batch_size 128 \
    --eval_batch_size 32 \
    --max_steps 3000 \
    --mlm_probability 0.4 \
    --logging_steps 100 \
    --save_steps 500 \
    --hidden_size 768 \
    --num_layers 12 \
    --num_head 8 \
    --num_workers 3 \
    --intermediate_size 3072 \
    --dropout_prob 0.1 \
    --warmup_ratio 0.1 \
    --eps 1e-12 \
    --save_dir ./exps/seed1234

## Model 3
mkdir ./exps/seed95
python train.py --data_dir ./data \
    --meta_data_file meta_data.csv \
    --profile_data_file profile_data.csv \
    --history_data_file history_data.csv \
    --seed 95 \
    --epochs 50 \
    --max_length 50 \
    --learning_rate 1e-4 \
    --weight_decay 1e-3 \
    --train_batch_size 128 \
    --eval_batch_size 32 \
    --max_steps 3000 \
    --mlm_probability 0.4 \
    --logging_steps 100 \
    --save_steps 500 \
    --hidden_size 768 \
    --num_layers 12 \
    --num_head 8 \
    --num_workers 3 \
    --intermediate_size 3072 \
    --dropout_prob 0.1 \
    --warmup_ratio 0.1 \
    --eps 1e-12 \
    --save_dir ./exps/seed95

# Prediction
python predict.py  --data_dir ./data \
    --meta_data_file meta_data.csv \
    --submission_file sample_submission.csv \
    --profile_data_file profile_data.csv \
    --history_data_file history_data.csv \
    --max_length 50 \
    --eval_batch_size 32 \
    --model_path exps/seed42/checkpoint-3000.pt \
    --num_workers 3 \
    --hidden_size 768 \
    --num_layers 12 \
    --num_head 8 \
    --intermediate_size 3072 \
    --dropout_prob 0.1 \
    --eps 1e-12 \
    --output_file ./results/2022_11_17_1.csv

# Ensemble
python ensemble.py  --data_dir ./data \
    --meta_data_file meta_data.csv \
    --submission_file sample_submission.csv \
    --profile_data_file profile_data.csv \
    --history_data_file history_data.csv \
    --max_length 50 \
    --eval_batch_size 32 \
    --num_workers 3 \
    --hidden_size 768 \
    --num_layers 12 \
    --num_head 8 \
    --intermediate_size 3072 \
    --dropout_prob 0.1 \
    --eps 1e-12 \
    --output_file ./results/2022_11_17_2.csv
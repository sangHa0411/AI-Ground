# Training with Evaluation

# Bert
python train.py --data_dir ./data \
    --seed 42 \
    --meta_data_file meta_data.csv \
    --profile_data_file profile_data.csv \
    --history_data_file history_data.csv \
    --max_length 100 \
    --do_eval True \
    --learning_rate 1e-4 \
    --weight_decay 1e-2 \
    --train_batch_size 280 \
    --eval_batch_size 32 \
    --max_steps 3200 \
    --mlm_probability 0.5  \
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
mkdir ./exps/seed1
python train.py --data_dir ./data \
    --meta_data_file meta_data.csv \
    --profile_data_file profile_data.csv \
    --history_data_file history_data.csv \
    --seed 1 \
    --max_length 100 \
    --learning_rate 1e-4 \
    --weight_decay 1e-2 \
    --train_batch_size 280 \
    --max_steps 3200 \
    --mlm_probability 0.5 \
    --logging_steps 100 \
    --save_steps 500 \
    --hidden_size 768 \
    --num_layers 12 \
    --num_head 8 \
    --num_workers 4 \
    --intermediate_size 3072 \
    --dropout_prob 0.1 \
    --warmup_ratio 0.1 \
    --eps 1e-12 \
    --save_dir ./exps/seed1

## Model 2
mkdir ./exps/seed2
python train.py --data_dir ./data \
    --meta_data_file meta_data.csv \
    --profile_data_file profile_data.csv \
    --history_data_file history_data.csv \
    --seed 2 \
    --max_length 100 \
    --learning_rate 1e-4 \
    --weight_decay 1e-2 \
    --train_batch_size 256 \
    --epochs 50 \
    --max_steps 3200 \
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
    --save_dir ./exps/seed2

## Model 3
mkdir ./exps/seed3
python train.py --data_dir ./data \
    --meta_data_file meta_data.csv \
    --profile_data_file profile_data.csv \
    --history_data_file history_data.csv \
    --seed 3 \
    --max_length 100 \
    --learning_rate 1e-4 \
    --weight_decay 1e-2 \
    --train_batch_size 256 \
    --epochs 50 \
    --max_steps 3200 \
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
    --save_dir ./exps/seed3

## Model 4
mkdir ./exps/seed4
python train.py --data_dir ./data \
    --meta_data_file meta_data.csv \
    --profile_data_file profile_data.csv \
    --history_data_file history_data.csv \
    --seed 4 \
    --max_length 100 \
    --learning_rate 1e-4 \
    --weight_decay 1e-2 \
    --train_batch_size 256 \
    --epochs 50 \
    --max_steps 3200 \
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
    --save_dir ./exps/seed4

## Model 5
mkdir ./exps/seed5
python train.py --data_dir ./data \
    --meta_data_file meta_data.csv \
    --profile_data_file profile_data.csv \
    --history_data_file history_data.csv \
    --seed 5 \
    --max_length 100 \
    --learning_rate 1e-4 \
    --weight_decay 1e-2 \
    --train_batch_size 256 \
    --epochs 50 \
    --max_steps 3200 \
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
    --save_dir ./exps/seed5

# Prediction
python predict.py  --data_dir ./data \
    --meta_data_file meta_data.csv \
    --submission_file sample_submission.csv \
    --profile_data_file profile_data.csv \
    --history_data_file history_data.csv \
    --max_length 100 \
    --eval_batch_size 32 \
    --model_path exps/seed1/checkpoint-3200.pt \
    --num_workers 4 \
    --hidden_size 768 \
    --num_layers 12 \
    --num_head 8 \
    --intermediate_size 3072 \
    --dropout_prob 0.1 \
    --eps 1e-12 \
    --output_file ./results/2022_11_28_0.csv

# Ensemble
python ensemble.py  --data_dir ./data \
    --meta_data_file meta_data.csv \
    --submission_file sample_submission.csv \
    --profile_data_file profile_data.csv \
    --history_data_file history_data.csv \
    --max_length 100 \
    --eval_batch_size 32 \
    --num_workers 3 \
    --hidden_size 768 \
    --num_layers 12 \
    --num_head 8 \
    --intermediate_size 3072 \
    --dropout_prob 0.1 \
    --eps 1e-12 \
    --output_file ./results/2022_11_25_0.csv
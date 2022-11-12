# Training with Evaluation
python train.py --data_dir ./data \
    --meta_data_file meta_data.csv \
    --profile_data_file profile_data.csv \
    --history_data_file history_data.csv \
    --max_length 200 \
    --do_eval True \
    --learning_rate 2e-4 \
    --weight_decay 1e-3 \
    --train_batch_size 128 \
    --eval_batch_size 32 \
    --epochs 50 \
    --mlm_probability 0.2 \
    --num_workers 6 \
    --logging_steps 100 \
    --eval_steps 500 \
    --hidden_size 768 \
    --num_layers 6 \
    --num_head 8 \
    --intermediate_size 3072 \
    --dropout_prob 0.1 \
    --warmup_ratio 0.1 \
    --eps 1e-8 \
    --save_dir ./exp
    
# Full Training
python train.py --data_dir ./data \
    --meta_data_file meta_data.csv \
    --profile_data_file profile_data.csv \
    --history_data_file history_data.csv \
    --max_length 200 \
    --learning_rate 2e-4 \
    --weight_decay 1e-3 \
    --train_batch_size 128 \
    --eval_batch_size 32 \
    --epochs 50 \
    --save_steps 500 \
    --mlm_probability 0.2 \
    --num_workers 6 \
    --logging_steps 100 \
    --eval_steps 500 \
    --hidden_size 768 \
    --num_layers 6 \
    --num_head 8 \
    --intermediate_size 3072 \
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
    --model_path exps/checkpoint-3250.pt \
    --num_workers 6 \
    --hidden_size 768 \
    --num_layers 6 \
    --num_head 8 \
    --intermediate_size 3072 \
    --dropout_prob 0.1 \
    --warmup_ratio 0.1 \
    --eps 1e-8 \
    --output_file ./results/2022_11_12.csv
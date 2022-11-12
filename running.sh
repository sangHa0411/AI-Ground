python train.py --data_dir ./data \
    --meta_data_file meta_data.csv \
    --profile_data_file profile_data.csv \
    --history_data_file history_data.csv \
    --max_length 200 \
    --do_eval True \
    --learning_rate 1e-4 \
    --weight_decay 1e-3 \
    --train_batch_size 128 \
    --eval_batch_size 32 \
    --epochs 50 \
    --mlm_probability 0.15 \
    --num_workers 4 \
    --logging_steps 100 \
    --eval_steps 500 \
    --hidden_size 256 \
    --num_layers 6 \
    --num_head 8 \
    --intermediate_size 1024 \
    --dropout_prob 0.1 \
    --warmup_ratio 0.1 \
    --eps 1e-8 \
    --save_dir ./exp
    



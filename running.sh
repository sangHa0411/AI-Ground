python train.py --data_dir ./data \
    --meta_data_file meta_data.csv \
    --profile_data_file profile_data.csv \
    --history_data_file history_data.csv \
    --max_length 256 \
    --do_eval True \
    --learning_rate 1e-4 \
    --weight_decay 1e-3 \
    --train_batch_size 32 \
    --eval_batch_size 32 \
    --epochs 100 \
    --mlm_probability 0.15 \
    --num_workers 4 \
    --logging_steps 500 \
    --eval_steps 2000 \
    --hidden_size 512 \
    --num_layers 12 \
    --num_head 8 \
    --intermediate_size 2048 \
    --dropout_prob 0.1 \
    --warmup_ratio 0.1 \
    --eps 1e-12 \
    --save_dir ./exp
    



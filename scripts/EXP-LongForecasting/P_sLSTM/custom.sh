# add --individual for P-sLSTM
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=336

python -u run_longExp.py \
    --is_training 1 \
    --root_path ./dataset/ \
    --data_path train_processed.csv \
    --model_id weather_$seq_len'_'96 \
    --model P_sLSTM \
    --data custom \
    --features MS \
    --target target \
    --seq_len $seq_len \
    --pred_len 96 \
    --label_len 24 \
    --des 'Exp' \
    --itr 1 --batch_size 16 \
    --patch_size 56 --stride 56 \
    --num_blocks 2 \
    --channel 21 --embedding_dim 100 --num_heads 2 --conv1d_kernel_size 8 --group_norm_weight True \
    --dropout 0.1 --patience 5 >logs/LongForecasting/P_sLSTM_Custom_$seq_len'_'96.log

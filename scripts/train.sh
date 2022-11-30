seed=(2021 111 222 333 444 555 666 777 888 999)

for round in 3;
do
    CUDA_VISIBLE_DEVICES=3 python train.py \
    --seed ${seed[$round]} \
    --epoch 3 \
    --batch_size 1 \
    --model_name_or_path 'IDEA-CCNL/Erlangshen-DeBERTa-v2-320M-Chinese' \
    --warmup_ratio 0.5 \
    --max_length 256 \
    --output_dir ./log/${seed[$round]}
done


seed=(2021 111 222 333 444 555 666 777 888 999)

for round in 0;
do
    CUDA_VISIBLE_DEVICES=0 python train.py \
    --seed ${seed[$round]} \
    --epoch 1 \
    --batch_size 32 \
    --model_name_or_path 'IDEA-CCNL/Erlangshen-UniMC-DeBERTa-v2-1.4B-Chinese' \
    --warmup_ratio 0.5 \
    --max_length 256 
done

# epoch=20 is better for full training

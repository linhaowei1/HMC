seed=(2021 111 222 333 444 555 666 777 888 999)

for round in "111_epoch0" "222_epoch0" "333_epoch0" "2021_epoch1" "111_epoch1" "222_epoch1" "333_epoch1" "2021_epoch2" "111_epoch2" "222_epoch2" "333_epoch2";
do
    CUDA_VISIBLE_DEVICES=5 python predict.py \
    --batch_size 1 \
    --model_name_or_path '/home/haowei/haowei/NLP/log/'$round \
    --max_length 256
done

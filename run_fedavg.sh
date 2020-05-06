#!/usr/bin/env bash
python3  -u main.py --dataset=$1 --optimizer='fedavg'  \
            --learning_rate=$3 --num_rounds=$5 --clients_per_round=10 \
            --eval_every=1 --batch_size=10 \
            --num_epochs=20 \
            --model=$4 \
            --drop_percent=$2 \


#!/usr/bin/env bash
python3  -u main.py --dataset=$1 --optimizer='fedprox'  \
            --learning_rate=$4 --num_rounds=$6 --clients_per_round=10 \
            --eval_every=1 --batch_size=10 \
            --num_epochs=20 \
            --model=$5 \
            --drop_percent=$2 \
            --mu=$3 \

#!/bin/sh
DATA='taxi'
DATADIR='../data/Taxi/'
PRETRAIN=0

python main.py \
	--data_name $DATA \
	--data_path $DATADIR \
	--save_path './taxi_valid/' \
	--pretrain $PRETRAIN

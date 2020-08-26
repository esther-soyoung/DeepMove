#!/bin/sh
DATA='taxi'
DATADIR='../data/Taxi/'
PRETRAIN=0
CHK=4
EPOCH=16

python main.py \
	--data_name $DATA \
	--data_path $DATADIR \
	--save_path './taxi_valid/' \
	--pretrain $PRETRAIN \
	--load_checckpoint $CHK \
	--epoch_max $EPOCH

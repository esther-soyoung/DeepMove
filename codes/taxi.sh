#!/bin/sh
DATA='taxi'
DATADIR='../data/Taxi/'
PRETRAIN=0
CHK=6
EPOCH=0

python main.py \
	--data_name $DATA \
	--data_path $DATADIR \
	--save_path './taxi/' \
	--pretrain $PRETRAIN \
	--load_checkpoint $CHK \
	--epoch_max $EPOCH

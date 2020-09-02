#!/bin/sh
DATA='foursquare_la2'
DATADIR='../data/LA/'
PRETRAIN=1

python main.py \
	--data_name $DATA \
	--data_path $DATADIR \
	--save_path './la_valid/' \
	--pretrain $PRETRAIN \
	--learning_rate 0.00005

#!/bin/sh
DATA='foursquare2'
DATADIR='../data/Foursquare/'
PRETRAIN=1

python main.py \
	--data_name $DATA \
	--data_path $DATADIR \
	--save_path './ny_valid/' \
	--pretrain $PRETRAIN \
	--learning_rate 0.00005

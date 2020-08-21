#!/bin/sh
DATA='foursquare2'
DATADIR='../data/pickles/'
PRETRAIN=0

# L2
python main.py \
	--data_name $DATA \
	--data_path $DATADIR \
	--save_path './ny/l2_001/' \
	--pretrain $PRETRAIN \
	--L2 0.001 &&

python main.py \
	--data_name $DATA \
	--data_path $DATADIR \
	--save_path './ny/l2_0001/' \
	--pretrain $PRETRAIN \
	--L2 0.0001 &&

# learning rate
python main.py \
	--data_name $DATA \
	--data_path $DATADIR \
	--save_path './ny/lr_00005/' \
	--pretrain $PRETRAIN \
	--learning_rate 0.00005 &&

python main.py \
	--data_name $DATA \
	--data_path $DATADIR \
	--save_path './ny/lr_001/' \
	--pretrain $PRETRAIN \
	--learning_rate 0.001 &&

# lr step
python main.py \
	--data_name $DATA \
	--data_path $DATADIR \
	--save_path './ny/lrstep_3/' \
	--pretrain $PRETRAIN \
	--lr_step 3 &&

# lr decay
python main.py \
	--data_name $DATA \
	--data_path $DATADIR \
	--save_path './ny/lrdecay_0.05/' \
	--pretrain $PRETRAIN \
	--lr_decay 0.05 &&

python main.py \
	--data_name $DATA \
	--data_path $DATADIR \
	--save_path './ny/lrdecay_0.15/' \
	--pretrain $PRETRAIN \
	--lr_decay 0.15

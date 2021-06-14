#! /bin/bash

#######################################################
######### Script for Experiments w/ CIFAR10 ###########
#######################################################
gpu=2

## Train a Large Network (resnet18)
ex=0
nohup python -u main.py --train_flag --gpu_no ${gpu} --data CIFAR10 --batch_size 128 --epoch 300 --lr 0.1 --optim SGD --sgd_momentum 0.9 --num_workers 4 --weight_decay 0.0005 --save_path ./WEIGHTS/${ex} --model resnet18 --temperature 1 --distillation_weight 0.0 --scheduler MStepLR --lr_milestones 150 225 --print_interval 50 --valid_interval 20 > ./WEIGHTS/log/${ex}_train.log &


## Train a Small Network (without knowledge distillation)
ex=1
nohup python -u main.py --train_flag --gpu_no ${gpu} --data CIFAR10 --batch_size 128 --epoch 300 --lr 0.1 --optim SGD --sgd_momentum 0.9 --num_workers 4 --weight_decay 0.0005 --save_path ./WEIGHTS/${ex} --model 1 --temperature 1 --distillation_weight 0.0 --scheduler MStepLR --lr_milestones 150 225 --print_interval 50 --valid_interval 20 > ./WEIGHTS/log/${ex}_train.log &


## Check the effect of randomness with various random seeds
for seed in 10 20 30
do
    ex=$((ex + 1))
    nohup python -u main.py --train_flag --gpu_no ${gpu} --data CIFAR10 --batch_size 128 --epoch 300 --lr 0.1 --optim SGD --sgd_momentum 0.9 --num_workers 4 --weight_decay 0.0005 --save_path ./WEIGHTS/${ex} --model 1 --temperature 1 --distillation_weight 0.0 --scheduler MStepLR --lr_milestones 150 225 --print_interval 50 --valid_interval 20 --random_seed ${seed} > ./WEIGHTS/log/${ex}_train.log &
done


## Train a Small Network (with knowledge distillation)
ex=4
for T in 1 10 30 50 100
do
    ex=$((ex + 1))
    nohup python -u main.py --train_flag --gpu_no ${gpu} --data CIFAR10 --batch_size 128 --epoch 300 --lr 0.1 --optim SGD --sgd_momentum 0.9 --num_workers 4 --weight_decay 0.0005 --save_path ./WEIGHTS/${ex} --model 1 --temperature ${T} --distillation_weight 0.1 --scheduler MStepLR --lr_milestones 150 225 --print_interval 50 --valid_interval 20 --teacher_load ./WEIGHTS/0/check_point_300.pth > ./WEIGHTS/log/${ex}_train.log &
done

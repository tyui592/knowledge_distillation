
Distilling the Knowledge in a Neural Network
==
* Unofficial Pytorch Implementation of "Distilling the Knowledge in a Neural Network" with **CIFAR10 Dataset**
* Reference: [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531) **NIPS Workshop 2014**
* **Author**: `Minseong Kim`(tyui592@gmail.com)

Requirements
--
- torch (version: 1.7.1)
- torchvision (version: 0.8.2)

Usage
--

### 1. Train a Large Network
```bash
$ python main.py --train_flag --gpu_no 0 --data CIFAR10 --batch_size 128 --epoch 300 --lr 0.1 --optim SGD --sgd_momentum 0.9 --num_workers 4 --weight_decay 0.0005 --save_path ./WEIGHTS/0 --model resnet18 --temperature 1 --distillation_weight 0.0 --scheduler MStepLR --lr_milestones 150 225 --print_interval 50 --valid_interval 20
```

### 2. Train a Small Network (without knowledge distillation)
```bash
$ python main.py --train_flag --gpu_no 0 --data CIFAR10 --batch_size 128 --epoch 300 --lr 0.1 --optim SGD --sgd_momentum 0.9 --num_workers 4 --weight_decay 0.0005 --save_path ./WEIGHTS/1 --model 1 --temperature 1 --distillation_weight 0.0 --scheduler MStepLR --lr_milestones 150 225 --print_interval 50 --valid_interval 20
```

### 3. Train a Small Network (with knowledge distillation)
```bash
$ python main.py --train_flag --gpu_no 0 --data CIFAR10 --batch_size 128 --epoch 300 --lr 0.1 --optim SGD --sgd_momentum 0.9 --num_workers 4 --weight_decay 0.0005 --save_path ./WEIGHTS/2 --model 1 --temperature 30 --distillation_weight 0.1 --scheduler MStepLR --lr_milestones 150 225 --print_interval 50 --valid_interval 20 --teacher_load ./WEIGHTS/0/check_point_300.pth
```

Experimental Result
--
Top-1 accuracy of the trained model (300 epochs) with CIFAR10 Test Dataset.

Check more details in [scripts.sh](https://github.com/tyui592/knowledge_distillation/blob/master/scripts.sh) and [logs](https://drive.google.com/drive/folders/1R6Dhz5R19B2phRLqEg-Ic-h7AyJ6-aL7?usp=sharing)

#### Base Models
| Large Network | Small Network (w/o distillation) |
| --- | --- |
| 95.21 % | 73.26 % |

#### Effect of the Knowledge Distillation
Train the Small Network with the trained Large Network
| temperature | accuracy |
| --- | --- |
| 1 | 73.74 % |
| 10 | 74.62 % |
| 30 | 75.81 % |
| 50 | 75.65 % |
| 100 | 75.47 % |

#### Effect of the Random-ness (w/o knowledge distillation)
| random seed | accuracy |
| --- | --- |
| 777 | 73.26 % |
| 10 | 75.12 % |
| 20 | 73.91 % |
| 30 | 75.64 % |

Reference
--
* https://github.com/peterliht/knowledge-distillation-pytorch
* https://github.com/bearpaw/pytorch-classification

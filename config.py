import os
import torch
import random
import argparse
import numpy as np

from utils import save_pickle, load_pickle

def build_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu_no', type=int, default=0)
    parser.add_argument('--random_seed', type=int, default=777)
    parser.add_argument('--train_flag', action='store_true', default=False)

    # Data
    parser.add_argument('--data_path', type=str, default='./DATA')
    parser.add_argument('--data', type=str, default='CIFAR10', help="MNIST | CIFAR10 | CIFAR100")
    parser.add_argument('--num_class', type=int, default=10, help="dependent argument with 'data'")
    parser.add_argument('--num_workers', type=int, default=4)

    # Train Validate
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--print_interval', type=int, default=100, help="print log per every N iterations")
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--save_path', type=str, default='./WEIGHTS')
    parser.add_argument('--save_epoch', type=int, default=10, help="save model per every N epochs")
    parser.add_argument('--valid_interval', type=int, default=10, help="validate per every N epochs")

    # Network
    parser.add_argument('--model', type=str, default='1', help="1 | resnet18")
    parser.add_argument('--model_load', type=str, default=None)

    parser.add_argument('--teacher', type=str, default=None, help="dependent argument with 'teacher_load'")
    parser.add_argument('--teacher_load', type=str, default=None, help="path of the checkpoint file")

    # Distilling
    parser.add_argument('--temperature', type=int, default=1)
    parser.add_argument('--distillation_weight', type=float, default=0.3, help="0: no distillation, 1: use only soft-target")

    # Optimization
    parser.add_argument('--optim', type=str, default='Adam', help="SGD | Adam")
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--sgd_momentum', type=float, default=0.9)
    parser.add_argument('--adam_betas', type=float, nargs='+', default=(0.9, 0.999))

    parser.add_argument('--scheduler', type=str, default=None, help="StepLR | MStepLR")
    parser.add_argument('--lr_milestones', type=int, nargs='+', default=[150, 225])
    parser.add_argument('--lr_stepsize', type=int, default=150)
    parser.add_argument('--lr_gamma', type=float, default=0.1)

    return parser.parse_args()

def get_arguments():
    # arguments
    args = build_parser()
    args = set_dependent_arguments(args)

    # evaluate
    if not args.train_flag and args.model_load:
        # temp
        _model_load = args.model_load
        _train_flag = args.train_flag
        _gpu_no     = args.gpu_no

        # load prev arguments
        loaded_args = load_pickle(os.path.join(os.path.dirname(args.model_load), 'arguments.pickle'))

        args = loaded_args
        args.train_flag = _train_flag
        args.model_load = _model_load
        args.gpu_no     = _gpu_no

    print('*'*30+'\nArguments\n'+ '*'*30)
    for k, v in sorted(vars(args).items()):
        print("%s: %s"%(k, v))

    # set random seed to remove effect of the randomness
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark= False

    # set device
    device = torch.device('cuda:%d'%args.gpu_no) if args.gpu_no >= 0 else torch.device('cpu')

    # make save dir
    if args.train_flag:
        os.makedirs(args.save_path, exist_ok=True)

        # save argument
        save_pickle(os.path.join(args.save_path, 'arguments.pickle'), args)

    return args, device

def set_dependent_arguments(args):
    # set num class
    if args.data in ['MNIST', 'CIFAR10']:
        args.num_class = 10

    elif args.data in ['CIFAR100']:
        args.num_class = 100

    else:
        raise NotImplementedError("Not expected data: '%s'"%args.data)

    # set model architecture of the teahcer
    if args.teacher_load is not None:
        teacher_args = load_pickle(os.path.join(os.path.dirname(args.teacher_load), 'arguments.pickle'))
        args.teacher = teacher_args.model

    return args

import os

from loss import LossCalulcator
from data import get_dataloader
from optim import get_optimizer
from config import get_arguments
from network import load_network
from trainval import train, measure_accuracy

if __name__ == '__main__':
    # load argument
    args, device = get_arguments()

    # make a network instance
    network = load_network(args.model, args.model_load, args.num_class, device)

    # validation data loader
    val_loader = get_dataloader(train_flag = False, args = args)

    if args.train_flag:
        optimizer, scheduler = get_optimizer(network, args)

        # load a teacher for knowledge distillation
        teacher = None
        if args.teacher_load:
            teacher = load_network(args.teacher, args.teacher_load, args.num_class, device)

        # train data loader
        train_loader = get_dataloader(train_flag = True, args = args)

        # make loss calculator
        loss_calculator = LossCalulcator(args.temperature, args.distillation_weight).to(device)

        # train the network
        print("Training the network...")
        network = train(student         = network,
                        dataloader      = train_loader,
                        optimizer       = optimizer,
                        scheduler       = scheduler,
                        loss_calculator = loss_calculator,
                        device          = device,
                        args            = args,
                        teacher         = teacher,
                        val_dataloader  = val_loader)

    else:
        # evaluate the network
        print("Evalute the network...")
        measure_accuracy(network, val_loader, device)

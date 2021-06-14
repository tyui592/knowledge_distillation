import torch.optim as optim

def get_optimizer(models, args):
    if args.optim == 'SGD':
        optimizer = optim.SGD(models.parameters(),
                              lr        = args.lr,
                              momentum  = args.sgd_momentum,
                              weight_decay=args.weight_decay)

    elif args.optim == 'Adam':
        optimizer = optim.Adam(models.parameters(),
                               lr           = args.lr,
                               betas        = args.adam_betas,
                               weight_decay = args.weight_decay)
    else:
        raise NotImplementedError("Not expected optimizer: '%s'"%args.optim)

    scheduler = None
    if args.scheduler == None:
        pass

    elif args.scheduler == 'StepLR':
        scheduler = optim.lr_scheduler.StepLR(optimizer     = optimizer,
                                              step_size     = args.lr_stepsize,
                                              gamma         = args.lr_gamma)

    elif args.scheduler == 'MStepLR':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer     = optimizer,
                                                   milestones    = args.lr_milestones,
                                                   gamma         = args.lr_gamma)

    else:
        raise NotImplementedError("Not expected scheduler: '%s'"%args.scheduler)


    return optimizer, scheduler

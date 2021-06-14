import torch
import torchvision
import torchvision.transforms as transforms

def get_transformer(train_flag, args):
    if args.data in ['CIFAR10']:
        # Mean and Std of CIFAR10 Train data
        MEAN = [0.4914, 0.4822, 0.4465]
        STD  = [0.2471, 0.2435, 0.2616]

        if train_flag:
            transformer = transforms.Compose([
                        transforms.RandomCrop(32, padding=4),
                        transforms.RandomHorizontalFlip(),  # randomly flip image horizontally
                        transforms.ToTensor(),
                        transforms.Normalize(MEAN, STD)])
        else:
            transformer = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize(MEAN, STD)])

    else:
        raise NotImplementedError("Not expected data: '%s'"%args.data)

    return transformer

def get_dataloader(train_flag, args):
    transformer = get_transformer(train_flag, args)

    dataset = torchvision.datasets.__dict__[args.data](root         = args.data_path,
                                                       train        = train_flag,
                                                       download     = True,
                                                       transform    = transformer)

    dataloader = torch.utils.data.DataLoader(dataset        = dataset,
                                             batch_size     = args.batch_size,
                                             shuffle        = train_flag == True,
                                             num_workers    = args.num_workers,
                                             pin_memory     = True)
    return dataloader

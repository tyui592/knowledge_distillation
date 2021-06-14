import os
import time
import torch

def train(student, dataloader, optimizer, scheduler, loss_calculator, device, args, teacher=None, val_dataloader=None):
    best_accuracy = 0
    best_epoch = 0

    if teacher is not None:
        teacher.eval()

    for epoch in range(1, args.epoch+1):
        # train one epoch
        train_step(student, dataloader, optimizer, loss_calculator, device, args, epoch, teacher)

        # validate the network
        if (val_dataloader is not None) and (epoch % args.valid_interval == 0):
            accuracy = measure_accuracy(student, val_dataloader, device)
            if accuracy >= best_accuracy:
                best_accuracy = accuracy
                best_epoch = epoch

        # learning rate schenduling
        scheduler.step()

        # save check point
        if (epoch % args.save_epoch == 0) or (epoch == args.epoch):
            torch.save({'argument': args,
                        'epoch': epoch,
                        'state_dict': student.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'loss_log': loss_calculator.loss_log},
                        os.path.join(args.save_path, 'check_point_%d.pth'%epoch))

    print("Finished Training, Best Accuracy: %f (at %d epochs)"%(best_accuracy, best_epoch))
    return student

def train_step(student, dataloader, optimizer, loss_calculator, device, args, epoch, teacher=None):
    student.train()

    for i, (inputs, labels) in enumerate(dataloader, 1):
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = student(inputs.to(device))

        teacher_outputs = None
        if teacher is not None and args.distillation_weight > 0.0:
            with torch.no_grad():
                teacher_outputs = teacher(inputs.to(device))

        loss = loss_calculator(outputs          = outputs,
                               labels           = labels.to(device),
                               teacher_outputs  = teacher_outputs)
        loss.backward()
        optimizer.step()

        # print log
        if i % args.print_interval == 0:
            print("%s: Epoch [%3d/%3d], Iteration [%5d/%5d], Loss [%s]"%(time.ctime(),
                                                                         epoch,
                                                                         args.epoch,
                                                                         i,
                                                                         len(dataloader),
                                                                         loss_calculator.get_log()))
    return None

def measure_accuracy(model, dataloader, device):
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            outputs = model(images.to(device))
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels.to(device)).sum().cpu().item()

    print("Accuracy of the network on the 10000 test images: %f %%"%(100 * correct / total))

    return correct / total

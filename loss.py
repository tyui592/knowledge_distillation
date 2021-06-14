import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import defaultdict

class LossCalulcator(nn.Module):
    def __init__(self, temperature, distillation_weight):
        super().__init__()

        self.temperature         = temperature
        self.distillation_weight = distillation_weight
        self.loss_log            = defaultdict(list)
        self.kldiv               = nn.KLDivLoss(reduction='batchmean')

    def forward(self, outputs, labels, teacher_outputs=None):
        """
        Compute the knowledge-distillation (KD) loss given outputs, labels.
        Reference: https://github.com/peterliht/knowledge-distillation-pytorch
        """

        # Distillation Loss
        soft_target_loss = 0.0
        if teacher_outputs is not None and self.distillation_weight > 0.0:
            soft_target_loss = self.kldiv(F.log_softmax(outputs/self.temperature, dim=1), F.softmax(teacher_outputs/self.temperature, dim=1)) * (self.temperature ** 2)

        # Ground Truth Loss
        hard_target_loss = F.cross_entropy(outputs, labels, reduction='mean')

        total_loss = (soft_target_loss * self.distillation_weight) + hard_target_loss

        # Logging
        if self.distillation_weight > 0:
            self.loss_log['soft-target_loss'].append(soft_target_loss.item())

        if self.distillation_weight < 1:
            self.loss_log['hard-target_loss'].append(hard_target_loss.item())

        self.loss_log['total_loss'].append(total_loss.item())

        return total_loss

    def get_log(self, length=100):
        log = []
        # calucate the average value from lastest N losses
        for key in self.loss_log.keys():
            if len(self.loss_log[key]) < length:
                length = len(self.loss_log[key])
            log.append("%s: %2.3f"%(key, sum(self.loss_log[key][-length:]) / length))
        return ", ".join(log)

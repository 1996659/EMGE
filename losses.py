import torch
import torch.nn as nn
import torch.nn.functional as F


# class ATLoss(nn.Module):
#     def __init__(self):
#         super().__init__()

#     def forward(self, logits, labels):
#         # TH label
#         th_label = torch.zeros_like(labels, dtype=torch.float).to(labels)
#         th_label[:, 0] = 1.0
#         labels[:, 0] = 0.0

#         p_mask = labels + th_label
#         n_mask = 1 - labels

#         # Rank positive classes to TH
#         logit1 = logits - (1 - p_mask) * 1e30
#         loss1 = -(F.log_softmax(logit1, dim=-1) * labels).sum(1)

#         # Rank TH to negative classes
#         logit2 = logits - (1 - n_mask) * 1e30
#         loss2 = -(F.log_softmax(logit2, dim=-1) * th_label).sum(1)

#         # Sum two parts
#         loss = loss1 + loss2
#         loss = loss.mean()
#         return loss
   
# class ATLoss(nn.Module):
#     def __init__(self, pos_weight=0.5, neg_weight=1.0):
#         super(ATLoss, self).__init__()
#         self.pos_weight = pos_weight
#         self.neg_weight = neg_weight

#     def forward(self, logits, labels):
#         th_label = torch.zeros_like(labels, dtype=torch.float).to(labels)
#         th_label[:, 0] = 1.0
#         labels[:, 0] = 0.2

#         p_mask = labels + th_label
#         n_mask = 1 - labels

#         logit1 = logits - (1 - p_mask) *  logits.max().item()
#         loss1 = -(F.log_softmax(logit1, dim=-1) * labels * self.pos_weight).sum(1)

#         logit2 = logits - (1 - n_mask) * logits.max().item()
#         loss2 = -(F.log_softmax(logit2, dim=-1) * th_label * self.neg_weight).sum(1)

#         loss = loss1 + loss2
#         loss = loss.mean()
#         return loss
   
 
# class ATLoss(nn.Module):
#     def __init__(self, threshold_weight=1.0):
#         super(ATLoss, self).__init__()
#         self.threshold_weight = threshold_weight

#     def forward(self, logits, labels):
#         th_label = torch.zeros_like(labels, dtype=torch.float).to(labels)
#         th_label[:, 0] = self.threshold_weight
#         labels[:, 0] = 0.0

#         p_mask = labels + th_label
#         n_mask = 1 - labels

#         logit1 = logits - (1 - p_mask) * logits.max().item()  # 使用 logits.max() 动态调整阈值
#         loss1 = -(F.log_softmax(logit1, dim=-1) * labels).sum(1)

#         logit2 = logits - (1 - n_mask) * logits.max().item()
#         loss2 = -(F.log_softmax(logit2, dim=-1) * th_label).sum(1)

#         loss = loss1 + loss2
#         loss = loss.mean()
#         return loss




    # def get_label(self, logits, num_labels=-1):
    #     th_logit = logits[:, 0].unsqueeze(1)
    #     output = torch.zeros_like(logits).to(logits)
    #     mask = (logits > th_logit)
    #     if num_labels > 0:
    #         top_v, _ = torch.topk(logits, num_labels, dim=1)
    #         top_v = top_v[:, -1]
    #         mask = (logits >= top_v.unsqueeze(1)) & mask
    #     output[mask] = 1.0
    #     output[:, 0] = (output.sum(1) == 0.).to(logits)
    #     return output
import torch
import torch.nn as nn
import torch.nn.functional as F


class ATLoss(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        pos_weight = [1] * num_classes
        pos_weight[0] = 0.0
        self.BCE = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(pos_weight).cuda(), reduction='mean')
        self.EL = nn.CrossEntropyLoss()

    def forward(self, logits, labels):
        # margin_loss = self.get_margin_loss2(logits, labels)
        margin_loss = self.get_entropy_loss(logits, labels)
        # entropy_loss = self.get_entropy_loss(logits, labels)
        return margin_loss
        # return margin_loss + 0.05 * entropy_loss

    def get_margin_loss(self, logits, labels):
        # TH label
        th_label = torch.zeros_like(labels, dtype=torch.float).to(labels)
        th_label[:, 0] = 1.0
        labels[:, 0] = 0.0

        p_mask = labels + th_label
        n_mask = 1 - labels

        # Rank positive classes to TH
        logit3 = 1 - logits + logits[:, 0].unsqueeze(1)
        loss3 = (F.relu(logit3) * p_mask).sum(1)
        # Rank TH to negative classes
        logit4 = 1 + logits - logits[:, 0].unsqueeze(1)
        loss4 = (F.relu(logit4) * n_mask).sum(1)

        loss = loss3 + loss4
        loss = loss.mean()
        return loss

    def get_entropy_loss(self, logits, labels):
        relation_nums = logits.shape[-1]
        if relation_nums > 2:
            return torch.sum(self.BCE(logits, labels)) / relation_nums
            # return self.get_margin_loss(logits, labels)
        else:
            return self.EL(logits, labels.argmax(dim=1))

    def get_margin_loss2(self, logits, labels):
        # TH label
        th_label = torch.zeros_like(labels, dtype=torch.float).to(labels)
        th_label[:, 0] = 1.0
        labels[:, 0] = 0.0

        p_mask = labels + th_label
        n_mask = 1 - labels

        # Rank positive classes to TH
        logit1 = logits - (1 - p_mask) * 1e30
        loss1 = -(F.log_softmax(logit1, dim=-1) * labels).sum(1)

        # Rank TH to negative classes
        logit2 = logits - (1 - n_mask) * 1e30
        loss2 = -(F.log_softmax(logit2, dim=-1) * th_label).sum(1)

        # Sum two parts
        loss = loss1 + loss2
        loss = loss.mean()
        return loss

    def get_label(self, logits, num_labels=-1):
        th_logit = logits[:, 0].unsqueeze(1)
        output = torch.zeros_like(logits).to(logits)
        mask = (logits > th_logit)
        if num_labels > 0:
            top_v, _ = torch.topk(logits, num_labels, dim=1)
            top_v = top_v[:, -1]
            mask = (logits >= top_v.unsqueeze(1)) & mask
        output[mask] = 1.0
        output[:, 0] = (output.sum(1) == 0.).to(logits)
        return output

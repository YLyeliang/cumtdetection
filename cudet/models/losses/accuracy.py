import torch.nn as nn


def accuracy(pred,target,topk=1):
    """
    Compare pred with target label, to get the accuracy,
    if topk is tuple, return topk accuracy.
    """
    assert isinstance(topk,(int,tuple))
    if isinstance(topk,int):
        topk=(topk,)
        return_single=True
    else:
        return_single=False

    maxk=max(topk)
    _,pred_label=pred.topk(maxk,dim=1)  # get maxk greatest number index
    pred_label=pred_label.t()   # transform matrix.
    correct = pred_label.eq(target.view(1,-1).expand_as(pred_label))   # eq menas element-wise equality

    res=[]
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0,keepdim=True)
        res.append(correct_k.mul_(100./pred.size(0)))
    return res[0] if return_single else res


class Accuracy(nn.Module):

    def __init__(self,topk=(1,)):
        super().__init__()
        self.topk=topk

    def forward(self,pred,target):
        return accuracy(pred,target,self.topk)
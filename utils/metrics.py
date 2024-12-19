import torch


def entropy(probs):
    e = probs * torch.log(probs)
    e = e.sum(-1).neg()
    return e


def accuracy(logits, target):
    return (logits.argmax(-1) == target).sum() / target.numel()
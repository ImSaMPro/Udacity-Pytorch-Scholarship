from collections import namedtuple
from torch import nn


def test_model(model, criterion=nn.NLLLoss, image_size=224):
    Tester = namedtuple('Tester', ['model', 'criterion', 'image_size'])
    return Tester(model, criterion, image_size)
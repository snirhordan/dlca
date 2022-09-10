import torch

class Noise(object):
    def __init__(self, mean = 0., std = 0.01) -> None:
        self.mean = mean
        self.std = std
    def __call__(self, tensor, device : torch.device):
        return tensor + torch.randn(tuple(tensor.size()), device = device, requires_grad= False)*self.std + self.mean
    def __repr__(self) -> str:
        return self.__class__.__name__ + '(mean={0},std={0.1})'.format(self.mean,self.std)
import abc
import torch
from torch import Tensor


class Optimizer(abc.ABC):
    """
    Base class for optimizers.
    """

    def __init__(self, params):
        """
        :param params: A sequence of model parameters to optimize. Can be a
        list of (param,grad) tuples as returned by the Layers, or a list of
        pytorch tensors in which case the grad will be taken from them.
        """
        assert isinstance(params, list) or isinstance(params, tuple)
        self._params = params

    @property
    def params(self):
        """
        :return: A sequence of parameter tuples, each tuple containing
        (param_data, param_grad). The data should be updated in-place
        according to the grad.
        """
        returned_params = []
        for x in self._params:
            if isinstance(x, Tensor):
                p = x.data
                dp = x.grad.data if x.grad is not None else None
                returned_params.append((p, dp))
            elif isinstance(x, tuple) and len(x) == 2:
                returned_params.append(x)
            else:
                raise TypeError(f"Unexpected parameter type for parameter {x}")

        return returned_params

    def zero_grad(self):
        """
        Sets the gradient of the optimized parameters to zero (in place).
        """
        for p, dp in self.params:
            dp.zero_()

    @abc.abstractmethod
    def step(self):
        """
        Updates all the registered parameter values based on their gradients.
        """
        raise NotImplementedError()


class VanillaSGD(Optimizer):
    def __init__(self, params, learn_rate=1e-3, reg=0):
        """
        :param params: The model parameters to optimize
        :param learn_rate: Learning rate
        :param reg: L2 Regularization strength
        """
        super().__init__(params)
        self.learn_rate = learn_rate
        self.reg = reg

    def step(self):
        for p, dp in self.params:
            if dp is None:
                continue

            # TODO: Implement the optimizer step.
            #  Update the gradient according to regularization and then
            #  update the parameters tensor.
            # ====== YOUR CODE: ======
            dp += self.reg * p
            p -= self.learn_rate * dp 
            # ========================


class MomentumSGD(Optimizer):
    def __init__(self, params, learn_rate=1e-3, reg=0, momentum=0.9):
        """
        :param params: The model parameters to optimize
        :param learn_rate: Learning rate
        :param reg: L2 Regularization strength
        :param momentum: Momentum factor
        """
        super().__init__(params)
        self.learn_rate = learn_rate
        self.reg = reg
        self.momentum = momentum

        # TODO: Add your own initializations as needed.
        # ====== YOUR CODE: ======
        self.velocities = dict()
        for p, _ in self.params:
            self.velocities[p] = torch.zeros_like(p)
        # ========================

    def step(self):
        for p, dp in self.params:
            if dp is None:
                continue

            # TODO: Implement the optimizer step.
            # update the parameters tensor based on the velocity. Don't forget
            # to include the regularization term.
            # ====== YOUR CODE: ======
            moment = self.momentum * self.velocities[p]
            temp = dp + p * self.reg
            lr = self.learn_rate * temp
            self.velocities[p] = moment - lr
            p += self.velocities[p]
            # ========================


class RMSProp(Optimizer):
    def __init__(self, params, learn_rate=1e-3, reg=0, decay=0.99, eps=1e-8):
        """
        :param params: The model parameters to optimize
        :param learn_rate: Learning rate
        :param reg: L2 Regularization strength
        :param decay: Gradient exponential decay factor
        :param eps: Constant to add to gradient sum for numerical stability
        """
        super().__init__(params)
        self.learn_rate = learn_rate
        self.reg = reg
        self.decay = decay
        self.eps = eps

        # TODO: Add your own initializations as needed.
        # ====== YOUR CODE: ======
        ##self.prev = [ torch.zeros(dp.size()) for p, dp in self.params if dp is not None ]
        ##self.idx = 0 
        self.r = dict()
        for p, _ in self.params:
            self.r[p] = torch.zeros_like(p)
        
        # ========================

    def step(self):
        for p, dp in self.params:
            if dp is None:
                continue

            # TODO: Implement the optimizer step.
            # Create a per-parameter learning rate based on a decaying moving
            # average of it's previous gradients. Use it to update the
            # parameters tensor.
            # ====== YOUR CODE: ======
            ##dp += self.reg * p

            ##self.prev[self.idx] = self.decay * self.prev[self.idx] + (1-self.decay) * torch.square(dp)
            ##temp = self.prev[self.idx].add_( self.eps )
            ##temp = temp.pow_(-1)
            ##temp = torch.sqrt( temp )
            ##temp = torch.mul( temp, self.learn_rate )
            ##temp = torch.mul( temp, dp )
            
            ##p = torch.sub( p, temp )#includes learn rate
            ##self.idx += 1
            ##if self.idx == len(self.prev):
                ##self.idx = 0
            first_value = self.decay * self.r[p]
            decay_compliment = 1 - self.decay
            self.r[p] =  first_value + decay_compliment*((dp + p * self.reg) ** 2)
            sqrt = torch.sqrt(self.r[p] + self.eps)
            p -= self.learn_rate * (dp + p * self.reg)/sqrt
            # ========================

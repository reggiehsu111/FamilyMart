import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class BaseModel(nn.Module, ABC):
    """
    Base class for all models.
    """
    @abstractmethod
    def forward(self, *inputs):
        return NotImplemented

    def __str__(self):
        """
        Prints the model and the number of trainable parameters.
        """
        trainable_params = filter(lambda p: p.requires_grad, self.parameters())

        n_params = sum(
            [torch.prod(torch.tensor(p.size())) for p in trainable_params])

        return (super().__str__() +
                '\nTrainable parameters: {}'.format(n_params))

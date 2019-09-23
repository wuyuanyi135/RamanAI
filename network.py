from typing import Any

import torch
from itertools import islice


def window(seq, n=2):
    "Returns a sliding window (of width n) over data from the iterable"
    "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result

class RamanAINetwork(torch.nn.Module):
    def __init__(self, network_structure: list):
        """

        :param network_structure: first element should be equal to input size.
        """
        super().__init__()
        self.layers = []

        for in_size, out_size in window(network_structure, 2):
            self.layers.append(torch.nn.Linear(in_size, out_size))
            self.layers.append(torch.nn.ReLU())

        self.layers.pop()
        self.layers = torch.nn.ModuleList(self.layers)

    def forward(self, input):
        for l in self.layers:
            input = l(input)
        return input




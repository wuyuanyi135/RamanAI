import os

import scipy.io as sio

import torch
import torch.utils.data
from yacs.config import CfgNode

import utils
from dataset import RamanDataset


def make_dataset(data: torch.Tensor, target: torch.Tensor = None, normalize=True):
    dataset = RamanDataset(data, target, normalize)
    return dataset


def load_matlab_file(loader_node: CfgNode):
    data = sio.loadmat(loader_node.path)
    try:
        return make_dataset(data[loader_node.input_var_name], data[loader_node.target_var_name], loader_node.normalize)
    except:
        return make_dataset(data[loader_node.input_var_name], None, loader_node.normalize)


def load(loader_node: CfgNode, cfg_path: str = "."):
    loader_node = loader_node.clone()
    loader_node.path = utils.abs_or_offset_from(loader_node.path, cfg_path)

    adapter = loader_node.name
    if adapter == "matlab":
        dataset = load_matlab_file(loader_node)
        return dataset
    else:
        raise NotImplemented(f"{adapter} is not implemented")

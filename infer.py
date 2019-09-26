import argparse
import scipy.io as sio
import torch
from yacs import config
import numpy as np
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mat", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--var_name", required=True)
    parser.add_argument("--output_mat", required=True)
    parser.add_argument("--norm", required=True)

    args = parser.parse_args()

    var = sio.loadmat(args.mat)
    in_data = torch.tensor(var[args.var_name])

    with open(args.norm, "r") as f:
        cfg = config.load_cfg(f)
    input_mean = torch.tensor(cfg.input.mean)
    input_std = torch.tensor(cfg.input.std)

    target_mean = torch.tensor(cfg.target.mean)
    target_std = torch.tensor(cfg.target.std)

    in_data = (in_data - input_mean) / input_std

    with open(args.weights, "rb") as f:
        net = torch.load(f)
    net.eval()


    output = net(in_data) * target_std + target_mean

    sio.savemat(args.output_mat, {"output": output.detach().numpy()})


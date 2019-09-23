import os

from yacs import config

CN = config.CfgNode
cfg = CN(new_allowed=True)

cfg.skip_copy = False

cfg.train_data = CN()

cfg.train_data.loader = CN()
cfg.train_data.loader.name = "matlab"
cfg.train_data.loader.input_var_name = "concatenated_input"
cfg.train_data.loader.target_var_name = "concatenated_output"
cfg.train_data.loader.path = "../datasets/glu_conc.mat"
cfg.train_data.loader.batch_size = 128
cfg.train_data.loader.normalize = True

cfg.valid_data = CN()
cfg.valid_data.enabled = True
cfg.valid_data.loader = CN()
cfg.valid_data.loader.name = "matlab"
cfg.valid_data.loader.input_var_name = "concatenated_input"
cfg.valid_data.loader.target_var_name = "concatenated_output"
cfg.valid_data.loader.path = "../datasets/glu_conc.mat"
cfg.valid_data.loader.batch_size = 128
cfg.valid_data.loader.normalize = False

cfg.solver = CN()
cfg.solver.lr = 0.00001
cfg.solver.l2 = 1e-9
cfg.solver.momentum = 0.9
cfg.solver.epoch = 50
cfg.solver.device = "cpu"
cfg.solver.criterion = "torch.nn.MSELoss"

cfg.output = CN()
cfg.output.base_path = "../outputs/glu_conc"

cfg.network = CN()
cfg.network.structure = [3327, 10, 1]

cfg.train_stream_handler = "PlotSaveHandler"

if __name__ == '__main__':
    with open(os.path.join("configs", "default.yaml"), "w") as f:
        f.write(cfg.dump())

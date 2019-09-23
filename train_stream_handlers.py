import os
import sys
import csv
import rx
import torch
from rx.subject import Subject
from yacs.config import CfgNode
import utils
from network import RamanAINetwork


class PlotSaveHandler(rx.core.Observer):

    def __init__(self, cfg: CfgNode, cfg_dir="."):
        super(PlotSaveHandler, self).__init__()
        self.base_path = utils.abs_or_offset_from(cfg.output.base_path, cfg_dir)

        os.makedirs(self.base_path, exist_ok=True)
        os.makedirs(os.path.join(self.base_path, "weights"), exist_ok=True)

        self.e = []
        self.train_loss = []
        self.val_loss = []
        self.plot_sub = Subject()

    def on_next(self, value) -> None:
        e = value["epoch"]
        train_time = value["train_time"]
        val_time = value["valid_time"]
        train_loss = value["train_loss"]
        val_loss = value["valid_loss"]
        net: RamanAINetwork = value["net"]

        print(f"Epoch:{e}, train_loss={train_loss}, val_loss={val_loss}, train_time={train_time}, val_time={val_time}")
        self.e.append(e)
        self.train_loss.append(train_loss)
        self.val_loss.append(val_loss)

        # save
        torch.save(net, os.path.join(self.base_path, "weights", f"weights_{e}.pkl"))

        self.plot_sub.on_next((self.e, self.train_loss, self.val_loss))

    def on_error(self, error: Exception) -> None:
        raise error

    def on_completed(self) -> None:
        print("Completed")
        with open(os.path.join(self.base_path, f"loss.csv"), "w") as f:
            writer = csv.writer(f)
            writer.writerow(["Epoch", "Train", "Valid"])
            for e, t, v, in zip(self.e, self.train_loss, self.val_loss):
                writer.writerow([e, t, v])

        self.plot_sub.on_completed()
        sys.exit(0)

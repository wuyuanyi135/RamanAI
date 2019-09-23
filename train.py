import time
import torch
import torch.utils
import torch.utils.data
from yacs.config import CfgNode

import utils
from network import RamanAINetwork
import rx
from rx import disposable
from rx import scheduler


def train_net(
        net: RamanAINetwork,
        train_data: torch.utils.data.Dataset,
        valid_data: torch.utils.data.Dataset,
        root_node: CfgNode
) -> rx.Observable:
    """

    :param net:
    :param train_data:
    :param valid_data:
    :param options:
    :return: Training loss, validation loss, time diff, and current net weights in one observable
    """

    def subscribe(observer: rx.typing.Observer, scheduler=None):
        solver_node = root_node.solver
        criterion = utils.str_to_obj(solver_node.criterion)()
        optimizer = torch.optim.SGD(
            net.parameters(),
            lr=solver_node.lr,
            weight_decay=solver_node.l2,
            momentum=solver_node.momentum
        )
        for e in range(solver_node.epoch):
            net.train()
            batch_start_time = time.time()
            loss_train = 0
            train_dataloader = torch.utils.data.DataLoader(train_data, root_node.train_data.loader.batch_size)
            for i, (input_data, target_data) in enumerate(train_dataloader):
                input_data = input_data.to(solver_node.device)
                output_data = net(input_data)

                optimizer.zero_grad()

                loss = criterion(output_data.to("cpu"), target_data)
                loss.backward()
                optimizer.step()
                loss_train += loss.item()
            batch_end_time = time.time()
            # done training step
            # start eval
            net.eval()
            loss_val = 0
            valid_dataloader = torch.utils.data.DataLoader(valid_data, root_node.valid_data.loader.batch_size)
            valid_start_time = time.time()
            for i, (input_data, target_data) in enumerate(valid_dataloader):
                with torch.no_grad():
                    input_data = input_data.to(solver_node.device)
                    output_data = net(input_data)
                loss = criterion(output_data.to("cpu"), target_data)
                loss_val += loss.item()
            valid_end_time = time.time()

            observer.on_next({
                "train_time": batch_end_time - batch_start_time,
                "valid_time": valid_end_time - valid_start_time,
                "train_loss": loss_train,
                "valid_loss": loss_val,
                "net": net,
                "epoch": e,
            })

        observer.on_completed()
        return disposable.Disposable()

    return rx.create(subscribe)

import argparse
import asyncio
import os
import sys
import shutil
import time

from yacs.config import CfgNode
import train_stream_handlers
import network
import train
import utils
from config import cfg
import dataloader
from matplotlib import pyplot as plt
from rx import operators
from rx import scheduler
from rx.scheduler.eventloop import AsyncIOThreadSafeScheduler

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="config file", required=True)
    args = parser.parse_args()

    args_config = os.path.abspath(args.config)
    cfg.merge_from_file(args_config)
    cfg_dir = os.path.dirname(args_config)

    print("Loading training data...")
    loader = cfg.train_data.loader
    train_dataset = dataloader.load(loader, cfg_dir)
    train_dataset.generate_original = False

    if cfg.valid_data.enabled:
        print("Loading validation data...")
        loader = cfg.valid_data.loader
        valid_dataset = dataloader.load(loader, cfg_dir)
        valid_dataset.normalize_with(
            train_dataset.input_mean,
            train_dataset.input_std,
            train_dataset.target_mean,
            train_dataset.target_std
        )
        valid_dataset.generate_original = False
    else:
        valid_dataset = None

    if not cfg.skip_copy:
        print("Creating reproducible working directory...")
        output_base_path = utils.abs_or_offset_from(cfg.output.base_path, cfg_dir)
        os.makedirs(output_base_path, exist_ok=True)

        print("Moving dataset to target directory...")
        train_path = utils.abs_or_offset_from(cfg.train_data.loader.path, cfg_dir)
        val_path = utils.abs_or_offset_from(cfg.valid_data.loader.path, cfg_dir)
        shutil.copy(train_path, output_base_path)
        shutil.copy(val_path, output_base_path)

        cfg_new = cfg.clone()
        cfg_new.train_data.loader.path = os.path.basename(cfg.train_data.loader.path)
        cfg_new.train_data.loader.path = os.path.basename(cfg.train_data.loader.path)
        cfg_new.valid_data.loader.path = os.path.basename(cfg.valid_data.loader.path)
        cfg_new.output.base_path = "."
        cfg_new.skip_copy = True

        print("Writing new configuration file...")
        with open(os.path.join(output_base_path, "config.yaml"), "w") as f:
            f.write(cfg_new.dump())

        print("Writing normallization file...")
        norm_node = CfgNode()
        norm_node.input = CfgNode()
        norm_node.input.mean = train_dataset.input_mean.tolist()
        norm_node.input.std = train_dataset.input_std.tolist()
        norm_node.target = CfgNode()
        norm_node.target.mean = train_dataset.target_mean.tolist()
        norm_node.target.std = train_dataset.target_std.tolist()
        with open(os.path.join(output_base_path, "norm.yaml"), "w") as f:
            f.write(norm_node.dump())
    else:
        print("Copy skipped")

    print("Building network...")
    net = network.RamanAINetwork(cfg.network.structure)

    print("Begin training...")
    stream = train.train_net(net, train_dataset, valid_dataset, cfg)
    handler = utils.str_to_obj(f"train_stream_handlers.{cfg.train_stream_handler}")

    # main loop scheduler
    loop = asyncio.new_event_loop()
    main_scheduler = AsyncIOThreadSafeScheduler(loop)
    s = scheduler.ThreadPoolScheduler()
    observer = handler(cfg, cfg_dir)

    stream.pipe(
        operators.subscribe_on(s),
    ).subscribe(observer)

    if observer.plot_sub:
        def plot_task(data):

            plt.plot(data[0], data[1], c="red")
            plt.plot(data[0], data[2], c="blue")
            plt.draw()
            plt.show(block=False)
            plt.pause(0.001)


        observer.plot_sub \
            .pipe(operators.observe_on(main_scheduler)) \
            .subscribe(plot_task, lambda e: loop.stop(), lambda: loop.stop())

    loop.run_forever()

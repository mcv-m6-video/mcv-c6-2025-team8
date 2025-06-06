import argparse
import torch
import os
import numpy as np
import random
import wandb
from torch.optim.lr_scheduler import (
    ChainedScheduler, LinearLR, CosineAnnealingLR)
import sys
from torch.utils.data import DataLoader
from tabulate import tabulate

# Local imports
from util.io import load_json, store_json
from util.eval_classification import evaluate
from dataset.datasets import get_datasets
from model_vince.model_classification import Model

from fvcore.nn import FlopCountAnalysis, parameter_count_table
import json

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--seed', type=int, default=1)
    return parser.parse_args()


def update_args(args, config):
    args.frame_dir = config['frame_dir']
    args.save_dir = config['save_dir'] + '/' + args.model
    args.store_dir = config['save_dir'] + '/' + "splits"
    args.labels_dir = config['labels_dir']
    args.store_mode = config['store_mode']
    args.task = config['task']
    args.batch_size = config['batch_size']
    args.clip_len = config['clip_len']
    args.dataset = config['dataset']
    args.epoch_num_frames = config['epoch_num_frames']
    args.feature_arch = config['feature_arch']
    args.learning_rate = config['learning_rate']
    args.num_classes = config['num_classes']
    args.num_epochs = config['num_epochs']
    args.warm_up_epochs = config['warm_up_epochs']
    args.only_test = config['only_test']
    args.device = config['device']
    args.num_workers = config['num_workers']
    return args


def get_lr_scheduler(args, optimizer, num_steps_per_epoch):
    cosine_epochs = args.num_epochs - args.warm_up_epochs
    print('Using Linear Warmup ({}) + Cosine Annealing LR ({})'.format(
        args.warm_up_epochs, cosine_epochs))
    return args.num_epochs, ChainedScheduler([
        LinearLR(optimizer, start_factor=0.01, end_factor=1.0,
                 total_iters=args.warm_up_epochs * num_steps_per_epoch),
        CosineAnnealingLR(optimizer,
            num_steps_per_epoch * cosine_epochs)])


def main(args):
    print('Setting seed to: ', args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    print(vars(args))  # Debugging step

    config_path = 'config/' + args.model + '.json'
    config = load_json(config_path)
    args = update_args(args, config)

    wandb.init(project="C6_W5", entity="C3_MCV_LGVP", config=config, name=args.model)

    ckpt_dir = os.path.join(args.save_dir, 'checkpoints')
    print(ckpt_dir)
    os.makedirs(ckpt_dir, exist_ok=True)

    classes, train_data, val_data, test_data = get_datasets(args)

    if args.store_mode == 'store':
        print('Datasets have been stored correctly! Re-run changing "mode" to "load" in the config JSON.')
        sys.exit('Datasets have correctly been stored! Stop training here and rerun with load mode.')
    else:
        print('Datasets have been loaded from previous versions correctly!')

    def worker_init_fn(id):
        random.seed(id + epoch * 100)

    train_loader = DataLoader(
        train_data, shuffle=False, batch_size=args.batch_size,
        pin_memory=True, num_workers=args.num_workers,
        prefetch_factor=(2 if args.num_workers > 0 else None),
        worker_init_fn=worker_init_fn
    )

    val_loader = DataLoader(
        val_data, shuffle=False, batch_size=args.batch_size,
        pin_memory=True, num_workers=args.num_workers,
        prefetch_factor=(2 if args.num_workers > 0 else None),
        worker_init_fn=worker_init_fn
    )

    # Load model for size evaluation
    model_test = Model(args=args)
    model_test._model.eval()
   
    # Dummy input: (batch_size, clip_len, channels, height, width)
    dummy_input = torch.randn(1, args.clip_len, 3, 224, 398).to(args.device)

    # Compute GFLOPs
    flop_analyzer = FlopCountAnalysis(model_test._model, dummy_input)
    gflops = flop_analyzer.total() / 1e9  # Convert to GFLOPs

    # Compute trainable and total parameters
    print(parameter_count_table(model_test._model))

    trainable_params = sum(p.numel() for p in model_test._model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model_test._model.parameters())

    # Print results
    print(f"GFLOPs: {gflops:.2f}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Total Parameters: {total_params:,}")

    # load real model
    model = Model(args=args)
    optimizer, scaler = model.get_optimizer({'lr': args.learning_rate})

    if not args.only_test:
        num_steps_per_epoch = len(train_loader)
        num_epochs, lr_scheduler = get_lr_scheduler(args, optimizer, num_steps_per_epoch)

        losses = []
        best_criterion = float('inf')
        epoch = 0

        print('START TRAINING EPOCHS')
        for epoch in range(epoch, num_epochs):
            train_loss = model.epoch(train_loader, optimizer, scaler, lr_scheduler=lr_scheduler)
            val_loss = model.epoch(val_loader)

            better = False
            if val_loss < best_criterion:
                best_criterion = val_loss
                better = True

            print('[Epoch {}] Train loss: {:0.5f} Val loss: {:0.5f}'.format(epoch, train_loss, val_loss))
            if better:
                print('New best mAP epoch!')

            losses.append({
                'epoch': epoch, 'train': train_loss, 'val': val_loss
            })

            wandb.log({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

            if args.save_dir is not None:
                os.makedirs(args.save_dir, exist_ok=True)
                store_json(os.path.join(args.save_dir, 'loss.json'), losses, pretty=True)

                if better:
                    torch.save(model.state_dict(), os.path.join(ckpt_dir, 'checkpoint_best.pt'))

    print('START INFERENCE')
    model.load(torch.load(os.path.join(ckpt_dir, 'checkpoint_best.pt')))

    ap_score = evaluate(model, test_data)

    table = []
    for i, class_name in enumerate(classes.keys()):
        table.append([class_name, f"{ap_score[i]*100:.2f}"])

    headers = ["Class", "Average Precision"]
    print(tabulate(table, headers, tablefmt="grid"))

    avg_map = np.mean(ap_score) * 100
    avg_table = [["Average", f"{avg_map:.2f}"]]
    headers = ["", "Average Precision"]
    print(tabulate(avg_table, headers, tablefmt="grid"))

    wandb.log({"Final mAP": avg_map})
    wandb.finish()

    print('CORRECTLY FINISHED TRAINING AND INFERENCE')

    # Dummy input: (batch_size, clip_len, channels, height, width)
    model.eval()
    dummy_input = torch.randn(1, config["clip_len"], 3, 224, 398).to(config["device"])

    # Compute FLOPs
    flop_analyzer = FlopCountAnalysis(model._model, dummy_input)
    gflops = flop_analyzer.total() / 1e9  # Convert to GFLOPs

    # Compute trainable and total parameters
    print(parameter_count_table(model._model))

    trainable_params = sum(p.numel() for p in model._model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model._model.parameters())

    # Print results
    print(f"GFLOPs: {gflops:.2f}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Total Parameters: {total_params:,}")
if __name__ == '__main__':
    main(get_args())

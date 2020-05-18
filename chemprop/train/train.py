import logging
import numpy as np
from typing import Callable

from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from tqdm import tqdm

from chemprop.args import TrainArgs
from chemprop.data import MoleculeDataLoader, MoleculeDataset
from chemprop.nn_utils import compute_gnorm, compute_pnorm, NoamLR

def train_dann(model: nn.Module,
                data_loader_train: MoleculeDataLoader,
                data_loader_test: MoleculeDataLoader,
                loss_func: Callable,
                optimizer: Optimizer,
                scheduler: _LRScheduler,
                args: TrainArgs,
                n_iter: int = 0,
                logger: logging.Logger = None,
                writer: SummaryWriter = None) -> int:
    """
    Trains a model for an epoch.

    :param model: Model.
    :param data_loader_source: A MoleculeDataLoader hosting training data.
    :param data_loader_test: A MoleculeDataLoader hosting testing data.
    :param loss_func: Loss function.
    :param optimizer: An Optimizer.
    :param scheduler: A learning rate scheduler.
    :param args: Arguments.
    :param n_iter: The number of iterations (training examples) trained on so far.
    :param logger: A logger for printing intermediate results.
    :param writer: A tensorboardX SummaryWriter.
    :return: The total number of iterations (training examples) trained on so far.
    """
    debug = logger.debug if logger is not None else print
    
    model.train()
    loss_sum, iter_count = 0, 0

    steps = 0
    data_test_iter = iter(data_loader_test)

    total_steps = len(data_loader_train)
    for batch in tqdm(data_loader_train, total=total_steps):

        current_epoch = int(n_iter/len(data_loader_train._dataset))
        p = float(steps + current_epoch  * total_steps) / args.epochs / total_steps
        alpha = 2. / (1. + np.exp(-10 * p)) - 1

        # Prepare batch from train data
        batch: MoleculeDataset
        mol_batch, features_batch, target_batch = batch.batch_graph(), batch.features(), batch.targets()
        mask = torch.Tensor([[x is not None for x in tb] for tb in target_batch])
        targets = torch.Tensor([[0 if x is None else x for x in tb] for tb in target_batch])
        domain_labels = torch.zeros(len(batch), 1)

        # Run model over train data
        model.zero_grad()
        preds, domain_preds = model(mol_batch, features_batch, alpha)

        # Move tensors to correct device
        mask = mask.to(preds.device)
        targets = targets.to(preds.device)
        class_weights = torch.ones(targets.shape, device=preds.device)
        domain_labels = domain_labels.to(domain_preds.device)

        if args.dataset_type == 'multiclass':
            targets = targets.long()
            loss = torch.cat([loss_func(preds[:, target_index, :], targets[:, target_index]).unsqueeze(1) for target_index in range(preds.size(1))], dim=1) * class_weights * mask
        else:
            loss = loss_func(preds, targets) * class_weights * mask
        # average predicton loss
        loss = loss.sum() / mask.sum()
        # average domain loss from train data
        loss += loss_func(domain_preds, domain_labels).sum() / domain_labels.shape[0]

        # Run model over test data
        if steps >= len(data_loader_test):
            steps = 0
            data_test_iter = iter(data_loader_test)

        batch_test = data_test_iter.next()
        batch_test: MoleculeDataset
        mol_batch_test, features_batch_test = batch_test.batch_graph(), batch_test.features()
        _, domain_preds_test = model(mol_batch_test, features_batch_test, alpha)

        domain_labels_test = torch.ones(len(batch_test), 1)
        domain_labels_test = domain_labels_test.to(domain_preds_test.device)
        
        # average domain loss from test data
        loss += loss_func(domain_preds_test, domain_labels_test).sum() / domain_labels_test.shape[0]

        # get loss sum over previous steps
        loss_sum += loss.item()
        iter_count += len(batch)

        # backpropagate
        loss.backward()
        optimizer.step()
        steps += 1

        if isinstance(scheduler, NoamLR):
            scheduler.step()

        n_iter += len(batch)

        # Log and/or add to tensorboard
        if (n_iter // args.batch_size) % args.log_frequency == 0:
            lrs = scheduler.get_lr()
            pnorm = compute_pnorm(model)
            gnorm = compute_gnorm(model)
            loss_avg = loss_sum / iter_count
            loss_sum, iter_count = 0, 0

            lrs_str = ', '.join(f'lr_{i} = {lr:.4e}' for i, lr in enumerate(lrs))
            debug(f'Loss = {loss_avg:.4e}, PNorm = {pnorm:.4f}, GNorm = {gnorm:.4f}, {lrs_str}')

            if writer is not None:
                writer.add_scalar('train_loss', loss_avg, n_iter)
                writer.add_scalar('param_norm', pnorm, n_iter)
                writer.add_scalar('gradient_norm', gnorm, n_iter)
                for i, lr in enumerate(lrs):
                    writer.add_scalar(f'learning_rate_{i}', lr, n_iter)

    return n_iter

def train(model: nn.Module,
          data_loader: MoleculeDataLoader,
          loss_func: Callable,
          optimizer: Optimizer,
          scheduler: _LRScheduler,
          args: TrainArgs,
          n_iter: int = 0,
          logger: logging.Logger = None,
          writer: SummaryWriter = None) -> int:
    """
    Trains a model for an epoch.

    :param model: Model.
    :param data_loader: A MoleculeDataLoader.
    :param loss_func: Loss function.
    :param optimizer: An Optimizer.
    :param scheduler: A learning rate scheduler.
    :param args: Arguments.
    :param n_iter: The number of iterations (training examples) trained on so far.
    :param logger: A logger for printing intermediate results.
    :param writer: A tensorboardX SummaryWriter.
    :return: The total number of iterations (training examples) trained on so far.
    """
    debug = logger.debug if logger is not None else print
    
    model.train()
    loss_sum, iter_count = 0, 0

    for batch in tqdm(data_loader, total=len(data_loader)):
        # Prepare batch
        batch: MoleculeDataset
        mol_batch, features_batch, target_batch = batch.batch_graph(), batch.features(), batch.targets()
        mask = torch.Tensor([[x is not None for x in tb] for tb in target_batch])
        targets = torch.Tensor([[0 if x is None else x for x in tb] for tb in target_batch])

        # Run model
        model.zero_grad()
        preds = model(mol_batch, features_batch)

        # Move tensors to correct device
        mask = mask.to(preds.device)
        targets = targets.to(preds.device)
        class_weights = torch.ones(targets.shape, device=preds.device)

        if args.dataset_type == 'multiclass':
            targets = targets.long()
            loss = torch.cat([loss_func(preds[:, target_index, :], targets[:, target_index]).unsqueeze(1) for target_index in range(preds.size(1))], dim=1) * class_weights * mask
        else:
            loss = loss_func(preds, targets) * class_weights * mask
        loss = loss.sum() / mask.sum()

        loss_sum += loss.item()
        iter_count += len(batch)

        loss.backward()
        optimizer.step()

        if isinstance(scheduler, NoamLR):
            scheduler.step()

        n_iter += len(batch)

        # Log and/or add to tensorboard
        if (n_iter // args.batch_size) % args.log_frequency == 0:
            lrs = scheduler.get_lr()
            pnorm = compute_pnorm(model)
            gnorm = compute_gnorm(model)
            loss_avg = loss_sum / iter_count
            loss_sum, iter_count = 0, 0

            lrs_str = ', '.join(f'lr_{i} = {lr:.4e}' for i, lr in enumerate(lrs))
            debug(f'Loss = {loss_avg:.4e}, PNorm = {pnorm:.4f}, GNorm = {gnorm:.4f}, {lrs_str}')

            if writer is not None:
                writer.add_scalar('train_loss', loss_avg, n_iter)
                writer.add_scalar('param_norm', pnorm, n_iter)
                writer.add_scalar('gradient_norm', gnorm, n_iter)
                for i, lr in enumerate(lrs):
                    writer.add_scalar(f'learning_rate_{i}', lr, n_iter)

    return n_iter

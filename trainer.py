import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
import os

"""
Trainer 训练器
"""


class Trainer:
    def __init__(self,
                 model: torch.nn.Module,
                 train_dataloader: DataLoader,
                 optimizer: torch.optim.Optimizer,
                 gpu_id: int) -> None:
        # rank
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.train_dataloader = train_dataloader
        self.optimizer = optimizer
        self.model = DDP(model, device_ids=[gpu_id], find_unused_parameters=True)

    def _run_batch(self, xs, ys):
        loss = self.model(xs, ys)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        print(f"loss:{loss.item():.3f}")

    def _run_epoch(self, epoch):
        batch_size = len(next(iter(self.train_dataloader))[0])
        print(f'[GPU: {self.gpu_id}] Epoch: {epoch} | Batchsize: {batch_size} | Steps: {len(self.train_dataloader)}')
        self.train_dataloader.sampler.set_epoch(epoch)
        for xs, ys in tqdm(self.train_dataloader):
            xs = xs.to(self.gpu_id)
            ys = ys.to(self.gpu_id)
            self._run_batch(xs, ys)

    def train(self, max_epoch: int):
        for epoch in range(max_epoch):
            self._run_epoch(epoch)

        # save model
        torch.save(self.model.state_dict(), os.path.join('model', "model_{}.pth".format(epoch)))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from tqdm import tqdm
import os
import logging
from inference import evaling

"""
Trainer 训练器
"""
logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self,
                 model: torch.nn.Module,
                 train_dataloader: DataLoader,
                 optimizer: torch.optim.Optimizer) -> None:
        # rank
        # self.gpu_id = gpu_id
        self.gpu_id = int(os.environ['LOCAL_RANK'])
        self.model = model.to(self.gpu_id)
        self.train_dataloader = train_dataloader
        self.optimizer = optimizer
        self.model = DDP(model, device_ids=[self.gpu_id], find_unused_parameters=True)

    def _run_batch(self, xs, ys):
        loss = self.model(xs, ys)
        loss.backward()

        # 同步梯度，确保参数更新的一致性
        # 注意：这一步在DDP v1中已经自动进行，但在DDP v2中需要手动进行
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(loss, op=dist.ReduceOp.SUM)
            loss.div_(dist.get_world_size())

        self.optimizer.step()
        self.optimizer.zero_grad()
        # print(f"loss:{loss.item():.3f}")
        return loss

    def _run_epoch(self, epoch):
        # training
        self.model.train()
        batch_size = len(next(iter(self.train_dataloader))[0])
        logger.info('this is training:')
        logger.info(
            f'[GPU: {self.gpu_id}] Epoch: {epoch} | Batchsize: {batch_size} | Steps: {len(self.train_dataloader)}')
        print(f'[GPU: {self.gpu_id}] Epoch: {epoch} | Batchsize: {batch_size} | Steps: {len(self.train_dataloader)}')
        self.train_dataloader.sampler.set_epoch(epoch)

        for xs, ys in self.train_dataloader:
            xs = xs.to(self.gpu_id)
            ys = ys.to(self.gpu_id)
            loss = self._run_batch(xs, ys)

            # 输出loss
            if self.gpu_id == 0:
                logger.info(f"loss:{loss.item():.3f}")

                print(f"loss:{loss.item():.3f}")

        # evl
        # self.model.load_state_dict(torch.load(os.path.join('model', "model_9.pth")), strict=False)
        self.model.eval()
        evaling(self.model)

    def train(self, max_epoch: int):
        for epoch in tqdm(range(max_epoch)):
            self._run_epoch(epoch)

        # save model
        torch.save(self.model.state_dict(), os.path.join('model', "model_{}.pth".format(epoch)))

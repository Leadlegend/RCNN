import torch
import numpy as np

from typing import Optional
from dataclasses import dataclass


@dataclass
class ImgBatch:
    datas: torch.Tensor  # [batch_size, img_size, img_size, 3]
    labels: Optional[torch.Tensor] = None

    def __getitem__(self, idx: int):
        if idx:
            return self.labels
        else:
            return self.datas

    def to(self, device):
        if self.labels is not None:
            return (self.datas.to(device), self.labels.to(device))
        else:
            return (self.datas.to(device), None)


def alex_collate_fn(batch, labeled: bool = True):
    if labeled:
        imgs, labels = trivial_collate_fn(batch)
        imgs, labels = torch.from_numpy(
            imgs).float().permute(0, 3, 1, 2), torch.LongTensor(labels)
        return ImgBatch(imgs, labels)
    else:
        imgs = trivial_collate_fn(batch)
        if isinstance(imgs, tuple):
            imgs = imgs[0]
        imgs = torch.from_numpy(imgs).float()
        return ImgBatch(imgs, None)


def ft_collate_fn(batch):
    return alex_collate_fn(batch, labeled=True)


def reg_collate_fn(batch):
    _, features, labels = trivial_collate_fn(batch)
    features, labels = torch.Tensor(
        features, dtype=torch.float32), torch.Tensor(labels).int().squeeze_()
    return ImgBatch(features, labels)


def trivial_collate_fn(batch):
    """
    Params:
        batch: List[Data]
    Return:
        mini_batch: List[Batch_Item]
        where Batch_Item is List[Data_Item]
    You should ensure that all the data in the batch of the same format
    which means that you can't collate mixed labeled/unlabeled data with this collate_fn
    """
    mini_batch = list()
    batch_size = len(batch)
    sdata = batch[0]
    for idx in range(len(sdata)):
        data_item = sdata[idx]
        try:
            data_size = [batch_size] + list(data_item.shape)
        except:
            data_size = [batch_size]
        batch_item = np.zeros(data_size)
        mini_batch.append(batch_item)

    for idx, data in enumerate(batch):
        for didx in range(len(data)):
            mini_batch[didx][idx] = data[didx]

    return tuple(mini_batch)

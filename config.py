import hydra
import torch.optim as opt

from functools import partial
from dataclasses import dataclass
from typing import Optional, List, Union
from omegaconf import DictConfig, OmegaConf
from hydra.core.config_store import ConfigStore
from model.criterions import *

cfg2opt = {
    "adam": partial(opt.Adam, betas=(0.9, 0.99), eps=1e-05),
    "sgd": partial(opt.SGD, momentum=0.9),
}
cfg2sch = {
    "None":
    None,
    "Step":
    partial(opt.lr_scheduler.StepLR, step_size=7, gamma=0.1),
    "Plateau":
    partial(
        opt.lr_scheduler.ReduceLROnPlateau,
        factor=0.93,
        mode='min',
        patience=10,
        cooldown=3,
        min_lr=1e-7,
    ),
}


def init_optimizer(cfg, model):
    opt, lr, sched = cfg.optimizer.lower(), cfg.lr, cfg.scheduler
    optimizer = cfg2opt[opt](params=model.parameters(), lr=lr)
    scheduler = cfg2sch.get(sched, None)
    if scheduler is not None:
        scheduler = scheduler(optimizer=optimizer)
    return optimizer, scheduler


@dataclass
class DatasetConfig:
    save_dir: str
    path: Union[List[str], str]
    info: Optional[dict] = None
    data_root: Optional[str] = None
    img_size: int = 227
    threshold: Optional[float] = 0.5
    reg_threshold: Optional[float] = 0.0
    batch_size: Optional[int] = 1
    pin: Optional[bool] = False
    shuffle: Optional[bool] = False
    workers: Optional[int] = 0
    lazy: Optional[bool] = False
    label: Optional[bool] = True
    context: Optional[int] = 0


@dataclass
class DataConfig:
    tokenizer: Optional[str]
    train: Optional[DatasetConfig]
    val: Optional[DatasetConfig]
    test: Optional[DatasetConfig]


@dataclass
class TrainerConfig:
    lr: float  # learning rate
    epoch: int  # epoch number
    device: str  # Cuda / cpu
    save_dir: str  # model checkpoint saving directory
    save_period: int = 5  # save one checkpoint every $save_period epoch
    ckpt: Optional[str] = None
    # model initialization or checkpoint resuming
    optimizer: Optional[str] = 'sgd'  # optimizer name
    scheduler: Optional[str] = 'Step'  # lr_scheduler name


@dataclass
class LoggerConfig:
    cfg_path: str
    save_dir: str = '.'


@dataclass
class ModelConfig:
    output_size: Optional[int]
    name: str = 'model'


@dataclass
class Config:
    name: str
    data: DataConfig
    trainer: TrainerConfig
    logger: LoggerConfig
    model: Optional[ModelConfig]


"""
@hydra.main(config_path='../config', config_name='base')
def main(cfg: Config):
    return cfg
"""


def args_util():
    """
        Set the template of experiment parameters (in hydra.config_store)
    """
    cs = ConfigStore.instance()
    cs.store(group='trainer', name='base_train', node=TrainerConfig)
    cs.store(group='model', name='base_model', node=ModelConfig)
    cs.store(group='data', name='base_train', node=DataConfig)
    cs.store(group='logger', name='base_base', node=LoggerConfig)


if __name__ == '__main__':
    args_util()

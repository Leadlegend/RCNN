import torch
import hydra

from logger import setup_logging
from trainer.svm import SvmTrainer
from data.datamodule import DataModule
from model import model_factory, criterion_factory
from config import args_util, init_optimizer


def train(cfg, idx: int, cnn):
    datamodule = DataModule(cfg=cfg.data, name=cfg.name, idx=idx)
    model = model_factory[cfg.name](cfg.model, class_idx=idx)
    criterion = criterion_factory[cfg.name]
    optim, sched = init_optimizer(cfg.trainer, model)
    trainer = SvmTrainer(feature_model=cnn, svm_model=model, config=cfg.trainer, device=cfg.trainer.device,
                         data_module=datamodule, criterion=criterion,
                         optimizer=optim, lr_scheduler=sched)
    trainer.train()


@hydra.main(config_path='./config', config_name='train_step3')
def main(configs):
    setup_logging(save_dir=configs.logger.save_dir,
                  log_config=configs.logger.cfg_path,
                  file_name=configs.name+'.log')
    torch.set_printoptions(precision=5)
    svm_num = configs.model.output_size - 1
    CNNModel = model_factory[configs.model.name](configs.model)
    CNNModel._init_weights(configs.trainer.ckpt, True)
    configs.trainer.ckpt = None
    for idx in range(1, svm_num + 1):
        train(cfg=configs, idx=idx, cnn=CNNModel)


if __name__ == '__main__':
    args_util()
    main()

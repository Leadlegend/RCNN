import os
import torch
import logging

from tqdm import tqdm
from torch.utils.data import SubsetRandomSampler, Subset, DataLoader

from .base import Trainer


class SvmTrainer(Trainer):
    def __init__(self,
                 feature_model,
                 svm_model,
                 criterion,
                 optimizer,
                 config,
                 device,
                 data_module,
                 lr_scheduler=None):
        super(SvmTrainer, self).__init__(model=feature_model,
                                         criterion=criterion, optimizer=optimizer,
                                         config=config, device=device, data_loader=None,
                                         lr_scheduler=lr_scheduler, log_step=4)
        self.data_module = data_module
        self.svm = svm_model.to(device)
        self.idx = self.svm.class_num
        self._overall_indices = None
        self._current_train_indices = None
        self.pos_size, self.neg_size = 0, 0
        if self.data_module.train_dataset is not None:
            self._init_data_loader()
        else:
            self.logger.error('No Training Data for SVM')
            raise ValueError

    def _init_data_loader(self):
        self._overall_indices = set(
            range(len(self.data_module.train_dataset)))
        self.pos_size = self.data_module.train_dataset.pos_num
        self.neg_size = pos_num * self.config.neg_prop
        pos_idx, neg_idx = self.data_module.train_dataset.init_sampler_indices(
            self.neg_size)
        self._current_train_indices = pos_idx + neg_idx
        self.logger.info(
            'Initial Train Dataset: %d Pos Samples: %d Neg Samples' % (self.pos_size, self.neg_size))
        sampler, val_sampler = self.build_subset_sampler(
            self._current_train_indices, require_val=True)
        self.data_loader = self.data_module.train_dataloader(
            sampler=sampler)
        self.val_data_loader = self.data_module.val_dataloader(
            sampler=val_sampler)
        self.logger.info('DataLoader Building Done, Start Training...')

    def _update_data_loader(self, indices):
        self._current_train_indices = list(
            set(self._current_train_indices).update(indices))
        self.neg_size = len(self._current_train_indices) - self.pos_size
        self.logger.info(
            'Current Train Dataset: %d Pos Samples: %d Neg Samples' % (self.pos_size, self.neg_size))
        sampler, val_sampler = self.build_subset_sampler(
            self._current_train_indices, require_val=True)
        self.data_loader = self.data_module.train_dataloader(sampler=sampler)
        self.val_data_loader = self.data_module.val_dataloader(
            sampler=val_sampler)
        self.logger.info('DataLoader Building Done, Start Training...')

    def _save_checkpoint(self, epoch):
        model_dict = self.svm.state_dict()
        filename = os.path.join(self.save_dir, 'svm-%s.pt' % self.idx)
        torch.save(model_dict, filename)
        self.logger.info("Saving checkpoint: {} ...".format(filename))

    def train(self):
        for epoch in range(self.start_epoch, self.epochs + 1):
            result = self._train_epoch(epoch)

            log = {'epoch': epoch}
            log.update(result)

            for key, value in log.items():
                self.logger.debug('{:8s}: {}'.format(str(key), value))

            hard_indices = self.hard_negative_mining()
            self._update_data_loader(hard_indices)
        self._save_checkpoint(self.epochs + 1)

    def _train_epoch(self, epoch):
        self.model.eval()
        self.svm.train()
        log = dict()
        len_epoch = len(self.data_loader)
        for batch_idx, (batch, _) in enumerate(tqdm(self.data_loader)):
            data, label = batch.to(self.device)
            self.optimizer.zero_grad()
            feature, final = self.model(data)
            output = self.svm(feature)
            loss = self.criterion(output, label)
            loss.backward()

            log.update({str(batch_idx): loss.item()})
            if batch_idx % self.log_step == 0:
                self.logger.info('Train Epoch: {} {}\t|\t Lr: {:.6f}\t|\tLoss: {:.6f}'.format(
                    epoch, self._progress(batch_idx), self.optimizer.param_groups[0]['lr'], loss.item()))
            if batch_idx >= len_epoch:
                break

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

    def hard_negative_mining(self):
        self.model.eval()
        self.svm.eval()
        num_corrects = 0
        hard_negative_indexes = set()
        self.logger.info('Start Hard Negative Mining...')
        with torch.no_grad():
            for batch_idx, (batch, data_indexes) in enumerate(tqdm(self.val_data_loader)):
                data, labels = batch.to(self.device)
                feature, final = self.model(data)
                outputs = self.svm(feature)
                _, preds = torch.max(output, dim=1)
                num_corrects += torch.sum(preds == labels)
                fp_mask = (preds == 1)
                tn_mask = (preds == 0)
                hard_negative_index = data_indexes[fp_mask]
                easy_negative_index = data_indexes[tn_mask]
                hard_negative_indexes |= hard_negative_index
        acc = num_corrects / len(self.val_data_loader)
        self.logger.info('Finished Validation.')
        self.logger.info('Validation Accuracy: %.5f, Found %d Hard Negative Samples' % (
            acc, len(hard_negative_indexes)))

        return hard_negative_indexes

    def get_remaining_indices(self, indices=None):
        if indices is None:
            indices = self._current_train_indices
        remain_indices = self._overall_indices - set(indices)
        return list(remain_indices)

    def build_subset_sampler(self, indices, require_val=False):
        train_sampler = SubsetRandomSampler(indices)
        if not require_val:
            return train_sampler, None
        val_indices = self.get_remaining_indices(indices)
        val_sampler = SubsetRandomSampler(val_indices)
        return train_sampler, val_sampler

    def build_subset_dataloader(self, indices, batch_size=None):
        dataset_split = Subset(self.data_module.train_dataset, indices)
        if batch_size is None:
            batch_size = self.data_module.cfg.train.batch_size
        return DataLoader(dataset_split, batch_size, shuffle=False, num_workers=2)

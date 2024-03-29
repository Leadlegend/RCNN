import os
import torch
import logging
import numpy as np

from tqdm import tqdm
from collections import defaultdict


def iou(pred_box, target_box):
    """
    :param pred_box: 大小为[4]
    :param target_box: 大小为[N, 4]
    :return: [N]
    """
    if len(target_box.shape) == 1:
        target_box = target_box[np.newaxis, :]

    xA = np.maximum(pred_box[0], target_box[:, 0])
    yA = np.maximum(pred_box[1], target_box[:, 1])
    xB = np.minimum(pred_box[2], target_box[:, 2])
    yB = np.minimum(pred_box[3], target_box[:, 3])
    # 计算交集面积
    intersection = np.maximum(0.0, xB - xA) * np.maximum(0.0, yB - yA)
    # 计算两个边界框面积
    boxAArea = (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1])
    boxBArea = (target_box[:, 2] - target_box[:, 0]) * \
        (target_box[:, 3] - target_box[:, 1])

    scores = intersection / (boxAArea + boxBArea - intersection)
    return scores


def nms(rect_list, score_list, threshold: float = 0.3):
    nms_rects, nms_scores = list(), list()
    rect_array = np.array(rect_list)
    score_array = np.array(score_list)

    idxs = np.argsort(score_array)[::-1]
    rect_array = rect_array[idxs]
    score_array = score_array[idxs]
    while len(score_array) > 0:
        # 添加分类概率最大的边界框
        nms_rects.append(rect_array[0])
        nms_scores.append(score_array[0])
        rect_array = rect_array[1:]
        score_array = score_array[1:]
        if len(score_array) <= 0:
            break

        iou_scores = iou(
            np.array(nms_rects[len(nms_rects) - 1]), rect_array)
        # 去除重叠率大于等于thresh的边界框
        idxs = np.where(iou_scores < threshold)[0]
        rect_array = rect_array[idxs]
        score_array = score_array[idxs]
    return nms_rects, nms_scores


class Tester:

    def __init__(self,
                 feature_model,
                 svm_models,
                 criterion,
                 config,
                 device,
                 dataset,
                 threshold=0.6):
        self.config = config
        self.device = device
        self.feature_model = feature_model.to(device)
        self.svm_models = [svm.to(device) for svm in svm_models]
        self.criterion = criterion
        self.threshold = threshold
        self.dataset = dataset
        self.len_epoch = len(self.dataset)
        self.checkpoint = config.ckpt
        self.save_dir = config.save_dir
        try:
            os.makedirs(self.save_dir)
        except:
            pass
        self.tokenizer = self.dataset.tokenizer

        # setup visualization writer instance
        self.logger = logging.getLogger('Tester')
        self._resume_checkpoint(self.checkpoint)

    def test(self):
        self._test_epoch()

    def _test_epoch(self):
        self.feature_model.eval()
        for svm in self.svm_models:
            svm.eval()
        with torch.no_grad():
            # for idx in tqdm(range(self.len_epoch)):
            for i in tqdm(range(2)):
                rect_dict = defaultdict(list)
                score_dict = defaultdict(list)
                imgs, rects, filename = self.dataset[i]
                data = torch.from_numpy(imgs).permute(
                    0, 3, 1, 2).float().to(self.device)
                feature, final = self.feature_model(data)
                preds = torch.argmax(final, dim=1).tolist()
                for idx, pred in enumerate(preds):
                    if pred != 0:
                        svm = self.svm_models[pred-1]
                        output = svm(feature[idx])
                        prob = torch.softmax(output, dim=0).tolist()[1]
                        if prob > self.threshold:
                            label = int(pred)
                            rect_dict[label].append(rects[idx])
                            score_dict[label].append(prob)
                for label in rect_dict.keys():
                    sorted_list = sorted(
                        zip(rect_dict[label], score_dict[label]), key=lambda x: x[-1], reverse=True)
                    rect_dict[label], score_dict[label] = [x[0]
                                                           for x in sorted_list], [x[-1] for x in sorted_list]
                    rect_dict[label], score_dict[label] = nms(
                        rect_dict[label], score_dict[label])
                self._generate_output(
                   rect_dict, score_dict, filename)

    def _generate_output(self, rect_dict, score_dict, filename):
        path = os.path.join(self.save_dir, filename)
        with open(path, 'w', encoding='utf-8') as f:
            for label in rect_dict.keys():
                assert int(label) > 0
                key = self.tokenizer(int(label))
                for rect, score in zip(rect_dict[label], score_dict[label]):
                    rect = [str(x) for x in rect[:4]]
                    line = list()
                    line.append(key)
                    line.append(str(score))
                    line += list(rect)
                    line = ' '.join(line) + '\n'
                    f.write(line)
            f.close()

    def _resume_checkpoint(self, ckpt_dir):
        resume_path = str(ckpt_dir)
        if not os.path.exists(ckpt_dir):
            self.logger.error("Bad checkpoint path: {}".format(resume_path))
            raise ValueError

        self.logger.info(
            "Loading Checkpoint of CNN Backend: {} ...".format(resume_path))
        resume_path = os.path.join(ckpt_dir, 'cnn.pt')
        checkpoint = torch.load(resume_path)
        if 'state_dict' in checkpoint:
            checkpoint = checkpoint['state_dict']
        try:
            self.feature_model.load_state_dict(checkpoint, strict=True)
        except Exception as e:
            self.logger.error('Bad checkpoint format.', stack_info=True)
            raise ValueError

        svm_files = ['svm%d.pt' % x for x in range(
            1, self.feature_model.cfg.output_size)]
        self.logger.info(
            "Loading Checkpoints of SVMs ...")
        for i, svm_file in enumerate(svm_files):
            svm_path = os.path.join(ckpt_dir, svm_file)
            checkpoint = torch.load(svm_path)
            if 'state_dict' in checkpoint:
                checkpoint = checkpoint['state_dict']
            try:
                self.svm_models[i].load_state_dict(checkpoint, strict=True)
            except Exception as e:
                self.logger.error('Bad checkpoint format.', stack_info=True)
                raise ValueError

        self.logger.info("Checkpoints loaded.")

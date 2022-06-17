# R-CNN PyTorch Reproduction

This is a reproduction project for 2014 CVPR paper 《Rich feature hierarchies for accurate object detection and semantic segmentation》 based on PyTorch. We reproduced R-CNN on `PASCAL VOC` dataset.

## Dependency

To install requiring packages, use

```bash
pip install -r requirements.txt
```

## Usage & Configuration

To train the R-CNN model, you can use scripts stored in `./script`, which is made up using configuration as below.
The configuration part for this repo used `hydra`, and the default configuration files is stored in `./config`. You can simply override any configuration by running.

```bash
python train_step2.py data.train.batch_size=512
```

which makes your size of mini-batch in training data 512.

To check your configuration before experiments, simply run:

```bash
python train_step2.py --cfg job
```

And your experimental logs and configurations will be stroed in `./outputs/some-directory`, you can set the specific directory by running:

```bash
python train_step2.py hydra.run.dir=./outputs/your-own-directory
```

## Evaluation

The evaluation part of this project is from [`Object-Detection-Metrics`](https://github.com/rafaelpadilla/Object-Detection-Metrics#asterisk) projects.

You can check the evaluation results by first running `./script/test.sh` and then runing `python evaluation/pascalvoc.py`.

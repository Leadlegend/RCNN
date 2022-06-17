run_dir='hydra.run.dir=./outputs/train_step3'
para='--cfg job'
CUDA_VISIBLE_DEVICES=0
nohup python train_step3.py trainer.ckpt=../../ckpt/step2/ckpt-epoch25.pt data.val.batch_size=2048 trainer.epoch=6 $run_dir > ./outputs/train_step3/train_step3_1.log & 

run_dir='hydra.run.dir=./outputs/train_step2'
para='--cfg job'
CUDA_VISIBLE_DEVICES=0
nohup python train_step2.py +kay=train $run_dir > ./outputs/train_step2/train_step2_2.log &

run_dir='hydra.run.dir=./outputs/train_step3'
para='--cfg job'
CUDA_VISIBLE_DEVICES=1
python train_step3.py trainer.ckpt=../../ckpt/step2/ckpt-epoch20.pt $run_dir  #> ./outputs/train_step3/train_step3.log 

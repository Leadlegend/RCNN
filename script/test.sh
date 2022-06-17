#export PYTHONWARNINGS='ignore:resource_tracker:UserWarning'
nohup python test.py hydra.run.dir=./outputs/test data.test.batch_size=1 > ./outputs/test/test.log &
cd test
python pascalvoc.py

#!/bin/bash

# pip install tensorboard

source activate tensorflow_p36

./leaf-run.py --epochs 300 CNN | tee CNN.log
# ./leaf-run.py --epochs 100 R50 | tee R50.log
./leaf-run.py --epochs 100 MNET | tee MNET.log

zip -r run.zip run.sh *.log leaf.*.h5 leaf-run.py leaf-setup.ipynb logs 

# sudo shutdown now

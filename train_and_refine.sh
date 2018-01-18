#!/bin/sh
python train_resnet.py --model_name=universal_resnet
python train_resnet.py --model_name=universal_resnet --refine_recording=0 --niter=50000
python train_resnet.py --model_name=universal_resnet --refine_recording=1 --niter=50000
python train_resnet.py --model_name=universal_resnet --refine_recording=2 --niter=50000
python train_resnet.py --model_name=universal_resnet --refine_recording=3 --niter=10000
python train_resnet.py --model_name=universal_resnet --refine_recording=4 --niter=50000

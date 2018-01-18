#!/bin/sh
python eval_resnet.py --model_name=universal_resnet
python eval_resnet.py --model_name=universal_resnet --refine_recording=0
python eval_resnet.py --model_name=universal_resnet --refine_recording=1
python eval_resnet.py --model_name=universal_resnet --refine_recording=2
python eval_resnet.py --model_name=universal_resnet --refine_recording=3
python eval_resnet.py --model_name=universal_resnet --refine_recording=4

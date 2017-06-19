#!/bin/sh
python predict.py --model_name=conservative_model_snapshot
python predict.py --model_name=conservative_model_snapshot --refine_recording=0
python predict.py --model_name=conservative_model_snapshot --refine_recording=1
python predict.py --model_name=conservative_model_snapshot --refine_recording=2
python predict.py --model_name=conservative_model_snapshot --refine_recording=3
python predict.py --model_name=conservative_model_snapshot --refine_recording=4

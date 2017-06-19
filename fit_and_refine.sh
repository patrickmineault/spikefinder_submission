#!/bin/sh
python fit_model.py --model_name=conservative_model_snapshot
python fit_model.py --model_name=conservative_model_snapshot --refine_recording=0 --niter=50000
python fit_model.py --model_name=conservative_model_snapshot --refine_recording=1 --niter=50000
python fit_model.py --model_name=conservative_model_snapshot --refine_recording=2 --niter=50000
python fit_model.py --model_name=conservative_model_snapshot --refine_recording=3 --niter=10000
python fit_model.py --model_name=conservative_model_snapshot --refine_recording=4 --niter=50000

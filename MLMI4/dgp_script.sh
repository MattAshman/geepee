#!/bin/bash
python2.7 run_regression.py --dataset boston --hidden_dims 2 2 2 --data_path ../../data/ 
python2.7 run_regression.py --dataset concrete --hidden_dims 2 2 2 --data_path ../../data/ 
python2.7 run_regression.py --dataset energy --hidden_dims 2 2 2 --data_path ../../data/ 
python2.7 run_regression.py --dataset kin8nm --hidden_dims  2 2 2 --data_path ../../data/ 
python2.7 run_regression.py --dataset naval --hidden_dims 2 2 2 --data_path ../../data/ 
python2.7 run_regression.py --dataset power --hidden_dims 2 2 2 --data_path ../../data/ 
python2.7 run_regression.py --dataset protein --hidden_dims 2 2 2 --data_path ../../data/ 
python2.7 run_regression.py --dataset wine_red --hidden_dims 2 2 2 --data_path ../../data/ 

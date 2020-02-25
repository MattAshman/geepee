import pdb
import sys
import os
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

import geepee.aep_models as aep
#import geepee.vfe_models as vfe
#import geepee.pep_models as pep
#import geepee.pep_models_tmp as pep_tmp
#from geepee.kernels import compute_kernel, compute_psi_weave
#import geepee.config as config
#import geepee.utils as utils

import numpy as np
import argparse
import time

from datasets import Datasets
from scipy.stats import norm

def main(args):
    num_layers = len(args.hidden_dims)
    datasets = Datasets(data_path=args.data_path)

    # Prepare output files
    outname1 = '../tmp/aep_' + args.dataset + '_' + str(num_layers) + '_'\
            + str(args.num_inducing) + '.rmse'
    if not os.path.exists(os.path.dirname(outname1)):
        os.makedirs(os.path.dirname(outname1))
    outfile1 = open(outname1, 'w')
    outname2 = '../tmp/aep_' + args.dataset + '_' + str(num_layers) + '_'\
            + str(args.num_inducing) + '.nll'
    outfile2 = open(outname2, 'w')
    outname3 = '../tmp/aep_' + args.dataset + '_' + str(num_layers) + '_'\
            + str(args.num_inducing) + '.time'
    outfile3 = open(outname3, 'w')

    running_err = 0
    running_loss = 0
    running_time = 0
    test_errs = np.zeros(args.splits)
    test_nlls = np.zeros(args.splits)
    test_times = np.zeros(args.splits)
    for i in range(args.splits):
        print('Split: {}'.format(i))
        print('Getting dataset...')
        data = datasets.all_datasets[args.dataset].get_data(i)
        X, Y, Xs, Ys, Y_std = [data[_] for _ in ['X', 'Y', 'Xs', 'Ys', 'Y_std']]
        
        dgp_model = aep.SDGPR(X, Y, args.num_inducing, args.hidden_dims)
        print('Training DGP model...')
        t0 = time.time()
        dgp_model.optimise(method='Adam', mb_size=args.batch_size,
                adam_lr=args.learning_rate, maxiter=args.iterations)
        t1 = time.time()
        test_times[i] = t1 - t0
        print('Time taken to train: {}'.format(t1 - t0))
        outfile3.write('Split {}: {}\n'.format(i+1, t1-t0))
        outfile3.flush()
        os.fsync(outfile3.fileno())
        running_time += t1 - t0

        # Minibatch test predictions
        means, vars = [], []
        test_batch_size = args.test_batch_size
        if len(Xs) > test_batch_size:
            for mb in range(-(-len(Xs) // test_batch_size)):
                m, v = dgp_model.predict_y(
                        Xs[mb*test_batch_size:(mb+1)*test_batch_size, :])
                means.append(m)
                vars.append(v)
        else:
            m, v = dgp_model.predict_y(Xs)
            means.append(m)
            vars.append(v)

        mean_ND = np.concatenate(means, 0)
        var_ND = np.concatenate(vars, 0)
        
        test_err = np.mean(Y_std * np.mean((Ys - mean_ND) ** 2.0) ** 0.5)
        test_errs[i] = test_err
        print('Average RMSE: {}'.format(test_err))
        outfile1.write('Split {}: {}\n'.format(i+1, test_err))
        outfile1.flush()
        os.fsync(outfile1.fileno())
        running_err += test_err

        test_nll = np.mean(norm.logpdf(Ys * Y_std, mean_ND * Y_std, 
            var_ND ** 0.5 * Y_std))
        test_nlls[i] = test_nll
        print('Average test log likelihood: {}'.format(test_nll))
        outfile2.write('Split {}: {}\n'.format(i+1, test_nll))
        outfile2.flush()
        os.fsync(outfile2.fileno())
        running_loss += test_nll
    
    outfile1.write('Average: {}\n'.format(running_err / args.splits))
    outfile1.write('Standard deviation: {}\n'.format(np.std(test_errs)))
    outfile2.write('Average: {}\n'.format(running_loss / args.splits))
    outfile2.write('Standard deviation: {}\n'.format(np.std(test_nlls)))
    outfile3.write('Average: {}\n'.format(running_time / args.splits))
    outfile3.write('Standard deviation: {}\n'.format(np.std(test_times)))
    outfile1.close()
    outfile2.close()
    outfile3.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--splits', default=20, type=int, 
            help='Number of cross-validation splits.')
    parser.add_argument('--data_path', default='../data/', 
            help='Path to datafile.')
    parser.add_argument('--dataset', help='Name of dataset to run.')
    parser.add_argument('--num_inducing', type=int, default=100, 
            help='Number of inducing input locations.')
    parser.add_argument('--learning_rate', type=float, default=0.005,
            help='Learning rate for optimiser.')
    parser.add_argument('--iterations', type=int, default=1000, 
            help='Number of training iterations.')
    parser.add_argument('--batch_size', type=int, default=50,
            help='Batch size to apply to training data.')
    parser.add_argument('--log_dir', default='./log/', 
            help='Directory log files are written to.')
    parser.add_argument('--test_batch_size', type=int, default=100, 
            help='Batch size to apply to test data.')
    parser.add_argument('--hidden_dims', type=int, default=[], nargs='+',
            help='Number of hidden dimensions, eg. 2 or 5 2.')
 
    args = parser.parse_args()
    main(args)

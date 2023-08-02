
from algorithms.model import train, test
import itertools
from distances import hamming_distance, levenshtein_distance
#from pathlib import Path
import os
import argparse
from .utils import parse_float_list

def main():

    parser = argparse.ArgumentParser(description='Anomaly detection via discrete optimization')
    parser.add_argument('--train_data', type=str, default='train_106', required=True,
                        help='available dataset for training (train_100, train_101, train_102, train_103, train_104, train_105, train_106)')
    parser.add_argument('--test_data', type=str, default='test_106', required=True,
                        help='available dataset for testing (test_100, test_101, test_103, test_104, test_105, test_106)')
    parser.add_argument('--Dist', choices= ('HD', 'LD'), default='LD', required=True,
                        help='use HD: hamming_distance, LD: levenshtein_distance')
    parser.add_argument('--cl', type=int, default='6',
                        help='correct labels (0, 1, 2, 3, 4, 5, 6)')
    parser.add_argument('--eps', type=parse_float_list, default='15',
                        help='any float value')
    parser.add_argument('--eps1', type=parse_float_list, default='1',
                        help='any float value')
    parser.add_argument('--eps2', type=parse_float_list, default='0.1',
                        help='any float value')

    args= parser.parse_args()

    output_dir='reports'
    os.makedirs(output_dir, exist_ok=True)
        
    if args.Dist == 'HD':
        Dist=hamming_distance
    elif args.Dist == 'LD':
        Dist=levenshtein_distance    

    # n denotes the number of states
    # Dist denotes distance function available: hamming_distance, levenshtein_distance
    # eps, eps1, eps2 are regularization parameters

    path_train = f"datasets/{args.train_data}.txt"
    path_test = f"datasets/{args.test_data}.txt"

    for n in range(2, 3):
        for i, j, k in itertools.product(args.eps, args.eps1, args.eps2):
            dfa1 = train(n, path_train, Dist, eps=i, eps1=j, eps2=k)
            test(path_test, args.cl, dfa1=dfa1)

if __name__ == "__main__":
    main()
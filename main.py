# main.py
from algorithms.model import train, test
import itertools
from distances import hamming_distance, levenshtein_distance
from pathlib import Path
import os

output_dir='reports'
os.makedirs(output_dir, exist_ok=True)
#data_folder = Path('C:/Users/bchan/Desktop/TUD/Thesis/')

# n denotes the number of states
# Dist denotes distance function available: hamming_distance, levenshtein_distance
# eps, eps1, eps2 are regularization parameters

path_train = "datasets/train_106.txt"
path_test = "datasets/test_106.txt"
diagram_path = 'reports/diagram_WP_LST_2.png'
model_path = 'reports/model_WP_LST_2.lp'
correct_label = 6
Dist = hamming_distance
eps, eps1, eps2 = [10], [1], [0.1]

for n in range(2, 3):
    for i, j, k in itertools.product(eps, eps1, eps2):
        dfa1 = train(n, path_train, Dist, model_path, diagram_path, eps=i, eps1=j, eps2=k)
        test(path_test, correct_label, dfa1=dfa1)

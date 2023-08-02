# AD_gurobi
Anomaly detection using DFA via Gurobi optimizer

# Deterministic Finite Automaton (DFA) Trainer and Tester

This project implements a trainer and tester for Deterministic Finite Automata (DFA) using the Gurobi optimization library. The trainer is used to learn a DFA model from a training dataset, and the tester evaluates the performance of the trained DFA on a test dataset. The project is organized into different files to improve code modularity and maintainability.

## Prerequisites


## Project Structure

The project is organized into the following files:

- `main.py`: The main script to run the training and testing process.
- `algorithms/model.py`: Contains functions for training the DFA using Gurobi and creating a diagram of the DFA.
- `preprocessing.py`: Contains functions for data preprocessing.
- `distances.py`: Contains distance functions like `hamming_distance`, `levenshtein_distance` as well as `Custom distances`.
- `utils.py`: collection of functions to make the other module shorter


## Usage
```
python main.py [-h] --train_data TRAIN_DATA --test_data TEST_DATA --Dist {HD,LD} [--cl CL]
               [--eps EPS] [--eps1 EPS1] [--eps2 EPS2]

optional arguments:
  -h, --help            show this help message and exit
  --train_data TRAIN_DATA
                        available dataset for training (train_100, train_101, train_102, train_103,
                        train_104, train_105, train_106)
  --test_data TEST_DATA
                        available dataset for testing (test_100, test_101, test_103, test_104,
                        test_105, test_106)
  --Dist {HD,LD}        use HD: hamming_distance, LD: levenshtein_distance
  --cl CL               correct labels (0, 1, 2, 3, 4, 5, 6)
  --eps EPS             list of float value
  --eps1 EPS1           list of float value
  --eps2 EPS2           list of float value
```
  

# AD_gurobi
Anomaly detection using DFA via Gurobi optimizer

# Deterministic Finite Automaton (DFA) Trainer and Tester

This project implements a trainer and tester for Deterministic Finite Automata (DFA) using the Gurobi optimization library. The trainer is used to learn a DFA model from a training dataset, and the tester evaluates the performance of the trained DFA on a test dataset. The project is organized into different files to improve code modularity and maintainability.

## Prerequisites

- Python 3.x
- Gurobi Optimizer (install via pip: `pip install gurobipy`)
- Pydot (install via pip: `pip install pydot`)
- Automata-lib (install via pip: `pip install automata-lib`)
- NumPy (install via pip: `pip install numpy`)
- scikit-learn (install via pip: `pip install scikit-learn`)
- itertools

## Project Structure

The project is organized into the following files:

- `main.py`: The main script to run the training and testing process.
- `algorithms/model.py`: Contains functions for training the DFA using Gurobi and creating a diagram of the DFA.
- `preprocessing.py`: Contains functions for data preprocessing.
- `distances.py`: Contains distance functions like `hamming_distance`, `levenshtein_distance` as well as Custom distances.


## Usage

1. Set the paths to the training and test dataset as well as to reports in `main.py`.

```python
path_train = "path/to/training_dataset.txt"
path_test = "path/to/test_dataset.txt"
diagram_path = "path to save model diagram"
model_path = "path to save model file"

2. Set the correct label (class) from 0-6 for Alfred dataset for the test and train dataset in main.py
3. Select the distance function (Hamming distance, Levenshtein distance)
4. Set the three regularization parameters 
5. Run the main.py script to train the DFA and evaluate its performance on the test dataset.
  

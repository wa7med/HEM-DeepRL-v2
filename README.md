# HEM-DeepRL-v2
This file provides the instructions to run the model.

## Install DEPENDENCIES
Some of the dependencies required to run the code, and some info about the environment (e.g., reward, actions, etc.).

 - tensorflow (>= 2.3)
 - numpy
 - Pandas
 - matplotlib

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install them.

```bash
C:\Project_PATH>pip install tensorflow
C:\Project_PATH>pip install numpy
C:\Project_PATH>pip install Pandas
C:\Project_PATH>pip install matplotlib
```

## Usage
There are 3 simple command line interface to run the code for training, testing and ploting:

### - Train a new model (this will save the model only at the end of the training):
```python
C:\Project_PATH>python main.py -train_model <number_of_Episodes>
```
(note that 300 Episodes should be enough)


### - Test a trained model (I've already provided a trained model in the "trained_models" folder)
```python
C:\Project_PATH>python main.py -test_model
```


### - Data plotting of the trained model:
You can plot one of the graph: "money_spent", "battery_charge", "reward_function"

```python
C:\Project_PATH>python main.py -plot_graph money_spent
```

```python
C:\Project_PATH>python main.py -plot_graph battery_charge
```

```python
C:\Project_PATH>python main.py -plot_graph reward_function
```

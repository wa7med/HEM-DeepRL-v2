This file provides the instruction, some of the dependencies required to run the code, and some info about the environment (e.g., reward, actions, etc.).
____________________________________________________________________________________________________

DEPENDENCIES ("pip install <name>"):

    - tensorflow (>= 2.3)
    - numpy
    - Pandas
    - matplotlib
____________________________________________________________________________________________________

HOW TO RUN:
Please use the terminal to run the model.
There is a simple command line interface to run the code both for training and for testing:
    - Train a new model (this will save the model only at the end of the training):
        - "python main.py -train_model <number_of_epochs>" (note that 300 epochs should be enough)
    - Test a trained model (I already provided a trained model in the "trained_models" folder)
        - "python main.py -test_model" (test the last trained model)
    - Plot data of the trained model:
        - "python main.py -plot_graph <graph_type>" (graph_type is one of "money_spent", "battery_charge", "reward_function")
		    python main.py -plot_graph money_spent
			python main.py -plot_graph battery_charge
			python main.py -plot_graph reward_function

When you test a model with the -test_model option, you will see the following information:
    Hello World Energy Manager!

    This years spent: 174.83 euros
    Total days with not enough charge on the EV: 0
    Total try to charge while EV is away: 2

where the total expenses for the year are printed, along with the days that the EV was not ready to go (i.e., was not charged) and the number of times the network tried to charge the EV while it was not at home (e.g., 2 actions on the 365*24 = 8760 total actions).
____________________________________________________________________________________________________

CLASSES:

- "main.py": runs the program and contains the different methods for test and training as detailed in the previous section.
- "plotter.py": contains different methods for plotting the results.
- "processed_data.py": uses Pandas to process .csv files
- "smart_home.py": is the Gym environment for the training
- "constant.py": defines the environment constants for the training (e.g., battery capacity, max possible recharge for hour, time windows to charge the EV, etc.)
____________________________________________________________________________________________________

TRAINING DETAILS:

- The training is performed on blocks of 30 days (each step is composed by 30 days).
- The battery is randomly initialize at each new episode within a "base" level according to the max capacity.
- I try to satisfy user's requests according to the following priority order: Home Battery -> PV -> SG
- Actions are taken at each hour; possible actions are:
    0. No actions
    1. Charge Home Battery with PV
    2. Charge Home Battery with SG
    3. Charge EV with PV
    4. Charge EV with SG
- I maximize the negate of the spent cost (in order to minimize the total cost); in addition the reward has some malus for undersired situations (i.e., EV is not charge in the morning or I try to charge the EV during the day).



I already provide a couple of graphs with the performance of the trained model in the "results" folder.
Market Interaction with DQN, Double DQN, and Transformer-based DQN

Welcome to our project! This repository demonstrates various deep reinforcement learning (RL) methods applied to interact with market data. Specifically, we utilize *DQN, **Double DQN, and a **Transformer-based DQN* approach to model and predict optimal trading actions in a dynamic market environment.

Project Structure

•⁠  ⁠data/: Contains market data used to train the models. The data is structured to simulate real market conditions for testing our RL algorithms.
  
•⁠  ⁠Brain.py: Contains the definition of the neural networks used in all three models (DQN, Double DQN, and Transformer-based DQN). This is where the architectures are designed.

•⁠  ⁠train.py: The main training script environments of "Discrete Allocation Levels" and "Binary Market Participation"

•⁠  ⁠train_smaller_steps.py: The training file for the "Increment Adjustments" action space

•⁠  ⁠big_run.py: This is the central script to run the entire project. It orchestrates the execution of the different models and simulates the market interactions.

•⁠  ⁠Run.py: A more focused script to execute a single run of a specific method (DQN or Double DQN). It's useful for testing one model at a time.

•⁠  ⁠run_with_transformer.py: This script is used to execute a single run with the Transformer-based DQN method. It focuses on running the transformer variant only.

•⁠  ⁠commands.txt: This file contains the necessary commands to execute the scripts. It provides the options and configurations to customize your run.

How to Run

To get started, simply follow the steps below:

1.⁠ ⁠Prepare the Environment: Ensure you have all necessary dependencies installed (likely from a requirements.txt or environment file).
2.⁠ ⁠Run the Project:
   - To run the full experiment, use big_run.py which handles all models.
   - To test a single model, use Run.py (for DQN or Double DQN) or run_with_transformer.py (for Transformer-based DQN).
   - The commands.txt file contains the exact command-line arguments required to run the scripts.

Quick Overview of Methods

•⁠  ⁠DQN (Deep Q-Network): Standard Q-learning with deep neural networks to approximate the Q-values.
  
•⁠  ⁠Double DQN: An improvement over DQN that helps reduce overestimation of Q-values, improving stability and performance.

•⁠  ⁠Transformer-based DQN: An advanced method using transformer networks to better capture long-term dependencies in the market data for more accurate decision-making.

We hope you find the project insightful, and feel free to explore or modify it as per your requirements.

Happy experimenting! 😄

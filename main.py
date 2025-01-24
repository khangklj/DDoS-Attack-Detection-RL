import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import PPO

from preprocessing import preprocess
from environment import NetworkTrafficEnv, EvalCallback   
from visualization import plot_avg_rewards, plot_evaluation_metrics

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='DDoS Defense Model Training')
    parser.add_argument('--data', type=str, default='./data/cicddos2019_dataset.csv', help='Path to the dataset file')
    parser.add_argument('--train_steps', type=int, default=10000, help='Number of training steps')
    parser.add_argument('--eval_freq', type=int, default=1000, help='Frequency of evaluation (in timesteps)')

    args = parser.parse_args()

    # Check if the data file exists
    if not os.path.exists(args.data):
        raise FileNotFoundError(f"Dataset file {args.data} not found.")

    df = pd.read_csv(args.data)
    preprocess(df)

    # Create the environment
    env = NetworkTrafficEnv(data=df)
    eval_env = NetworkTrafficEnv(data=df)  # Separate evaluation environment

    # Set up the model
    model = PPO("MlpPolicy", env, verbose=1)

    # Set up the evaluation callback
    eval_callback = EvalCallback(eval_env, eval_freq=args.eval_freq, verbose=1)

    # Train the model
    model.learn(total_timesteps=args.train_steps, callback=eval_callback)

    # Plot the average evaluation reward
    plot_avg_rewards(eval_callback.average_rewards, is_save_plot=True)

    # Plot the evaluation metrics
    plot_evaluation_metrics(eval_callback.eval_metrics, is_save_plot=True)    

    # Save the model
    if not os.path.exists("models"):
        os.makedirs("models")
    model.save("models/ddos_defense_model")   

if __name__ == '__main__':
    main()
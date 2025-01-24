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
    df = pd.read_csv("./data/cicddos2019_dataset.csv")
    preprocess(df)

    # Create the environment
    env = NetworkTrafficEnv(data=df)
    eval_env = NetworkTrafficEnv(data=df)  # Separate evaluation environment

    # Set up the model
    model = PPO("MlpPolicy", env, verbose=1)
    
    # Set up the evaluation callback
    eval_callback = EvalCallback(eval_env, eval_freq=1000, verbose=1)

    # Train the model
    model.learn(total_timesteps=10000, callback=eval_callback)

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
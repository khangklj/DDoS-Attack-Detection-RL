import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score

def plot_evaluation_metrics(metrics: list[tuple[float, float, float]], is_save_plot=False):   
    """
    Plots the evaluation metrics (precision, recall, and F1-score) over time.

    Parameters:
    - metrics (list of tuples): A list where each tuple contains three float values
      representing precision, recall, and F1-score at a specific evaluation step.
    - is_save_plot (bool, optional): If True, saves the plot as a PNG file in the 'logs'
      directory. Defaults to False.
    """

    precisions, recalls, f1_scores = zip(*metrics)
    precisions = list(precisions)
    recalls = list(recalls)
    f1_scores = list(f1_scores)

    # Plotting the evaluation metrics
    plt.figure(figsize=(15, 6))
    plt.plot(precisions, label='Precision', color='blue', marker='o')
    plt.plot(recalls, label='Recall', color='green', marker='s')
    plt.plot(f1_scores, label='F1 Score', color='orange', marker='^')
    plt.xlabel('Evaluation Steps (every 1000 timesteps)')
    plt.ylabel('Evaluation Metric')
    plt.title('Agent Evaluation During Training')
    plt.legend()
    plt.grid()

    # Save the plot as a PNG file
    if is_save_plot:
        if not os.path.exists("logs"):
            os.makedirs("logs")
        plt.savefig("logs/evaluation_metrics.png")

    plt.show()

def plot_avg_rewards(avg_rewards: list, is_save_plot=False):
    """
    Plots the average evaluation reward over time.

    Parameters:
    - avg_rewards (list of float): A list of average evaluation rewards at each evaluation step.
    - is_save_plot (bool, optional): If True, saves the plot as a PNG file in the 'logs' directory. Defaults to False.
    """   
    
    # Plotting the evaluation metrics
    plt.figure(figsize=(10, 5))
    plt.plot(avg_rewards, label='Average Evaluation Reward', color='blue')
    plt.xlabel('Evaluation Steps (every 1000 timesteps)')
    plt.ylabel('Average Reward')
    plt.title('Agent Evaluation During Training')
    plt.legend()
    plt.grid()

    # Save the plot as a PNG file
    if is_save_plot:
        if not os.path.exists("logs"):
            os.makedirs("logs")
        plt.savefig("logs/average_evaluation_reward.png")

    plt.show()
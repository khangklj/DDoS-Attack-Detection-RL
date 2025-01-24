import gym
from gym import spaces
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from sklearn.metrics import confusion_matrix

class NetworkTrafficEnv(gym.Env):
    def __init__(self, data):
        super(NetworkTrafficEnv, self).__init__()
        
        self.data = data
        self.max_steps = 100  # Set to the number of steps you want to allow per episode
        self.current_step = 0
        self.current_data = None  # Placeholder for the current sampled data
        
        # Define action space: 0 = Attack, 1 = Normal
        self.action_space = spaces.Discrete(2)
        
        # Define observation space: feature vectors
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(data.shape[1]-1,), dtype=np.float32)

    def reset(self):
        # Sample random 100 rows from the DataFrame
        self.current_data = self.data.sample(n=1000, random_state=np.random.randint(0, 10000)).reset_index(drop=True)
        self.current_step = 0
        return self.current_data.iloc[self.current_step, :-1].values  # Return the features of the first sampled row

    def step(self, action):
        # Get the true class for the current observation
        true_class = self.current_data.iloc[self.current_step]['Class']
        
        # Calculate reward based on classification outcome
        if action == 1 and true_class == 1:  # TP
            reward = 2
        elif action == 1 and true_class == 0:  # FP
            reward = -1
        elif action == 0 and true_class == 0:  # TN
            reward = 2
        elif action == 0 and true_class == 1:  # FN
            reward = -2
        
        # Move to the next step
        self.current_step += 1
        
        # Check if the episode is done
        done = self.current_step >= self.max_steps
        
        # Get the next observation
        next_observation = self.current_data.iloc[self.current_step, :-1].values if not done else np.zeros(self.observation_space.shape)
        
        return next_observation, reward, done, {}

# Custom callback for evaluation
class EvalCallback(BaseCallback):
    def __init__(self, eval_env, eval_freq, verbose=0):
        super(EvalCallback, self).__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.average_rewards = []
        self.cumulative_rewards = []
        self.eval_metrics = []

    def _on_step(self):
        if self.n_calls % self.eval_freq == 0:
            # Evaluate the model
            cumulative_reward = 0
            tp = fp = tn = fn = 0
            true_classes = []
            predicted_classes = []
            for _ in range(10):  # Evaluate over 10 episodes
                obs = self.eval_env.reset()
                done = False
                while not done:
                    action, _ = self.model.predict(obs)
                    obs, reward, done, _ = self.eval_env.step(action)
                    cumulative_reward += reward

                    # Update counts based on action and true class
                    true_class = self.eval_env.current_data.iloc[self.eval_env.current_step - 1]['Class']
                    true_classes.append(true_class)
                    predicted_classes.append(action)

                    if action == 1 and true_class == 1:
                        tp += 1
                    elif action == 1 and true_class == 0:
                        fp += 1
                    elif action == 0 and true_class == 0:
                        tn += 1
                    elif action == 0 and true_class == 1:
                        fn += 1

            average_reward = cumulative_reward / 10  # Average over 10 episodes
            self.average_rewards.append(average_reward)
            self.cumulative_rewards.append(cumulative_reward)

            # Calculate precision, recall, and F1-score
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            # Log the metrics
            self.eval_metrics.append((precision, recall, f1))            

            if self.verbose > 0:
                print(f"Evaluation at step {self.num_timesteps}: Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1:.2f}")

        return True

# DDoS-Attack-Detection-ML

This repository contains code for training a DDoS attack detection system. It uses reinforcement learning techniques to classify network traffic as DDoS or non-DDoS.

## Prerequisites

- numpy
- gym
- stable-baselines3
- shimmy
- pandas
- matplotlib
- scikit-learn
- seaborn

## Note

This model can only run on CPU.

## Installation

1. Clone the repository:

```
git clone https://github.com/khangklj/DDoS-Attack-Detection-RL.git
```

2. Change to working directory

```
cd DDoS-Attack-Detection-RL
```

3. Install the required packages:

```
pip install -r requirements.txt
```

## Usage

1. Download the [data](#data) and put in the
   csv file in this [**folder**](data)

2. Run the main script:

```
python main.py
```

Command line arguments:

- --data (str): Path to the dataset file (default: ./data/cicddos2019_dataset.csv)
- --train_steps (int): Number of training steps (default: 10000)
- --eval_freq (int): Frequency of evaluation (in timesteps) (default: 1000)
- --is_save_plot (bool): Save plots (default: False)
- --saved_model_dir (str): Directory to save the trained model (default: models)

## Dataset

The dataset used in this project is the [CIC-DDoS2019 Dataset](https://data.mendeley.com/datasets/ssnc74xm6r/1)

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).

## References:

- Talukder, Md A., and Md A. Uddin. "CIC-DDoS2019 Dataset." Mendeley, Mar 3 (2023).

- [Towers, M., Kwiatkowski, A., Terry, J., Balis, J. U., De Cola, G., Deleu, T., ... & Younis, O. G. (2024). Gymnasium: A standard interface for reinforcement learning environments. arXiv preprint arXiv:2407.17032.](https://arxiv.org/abs/2407.17032)

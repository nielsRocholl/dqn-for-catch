import re
import os
import numpy as np
import cv2

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def visualize(state: np.ndarray, enlarging_factor=5) -> None:
    """
    Visualize the state of the game using OpenCV
    :param state: state of the game
    :param enlarging_factor: factor to enlarge the state for better visualization
    :return: None
    """
    # Remove the batch dimension
    state = np.squeeze(state)
    size = 84 * enlarging_factor
    # get the most recent frame
    state = state[..., -1]
    resized_frame = cv2.resize(state, (size, size))
    color_frame = np.zeros((size, size, 3), dtype=np.uint8)

    # Set purple color for the ball and paddle
    color_frame[resized_frame == 1] = [255, 230, 0]

    # Set orange color for the background
    color_frame[resized_frame == 0] = [75, 0, 88]

    cv2.imshow('Catch Game', color_frame)
    cv2.waitKey(50)  # Adjust the delay between frames if needed (in milliseconds)


def destroy() -> None:
    """
    Destroy all windows
    :return: None
    """
    cv2.destroyAllWindows()


def plot_running_average_all_files():
    directory = 'performances/default_hyperparameter_tests/'
    window_size = 100
    all_scores = []
    plt.figure(figsize=(7, 5))

    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        data = pd.read_csv(file_path)
        scores = data['score']
        if len(scores) == 1500:
        # if filename == 'learningRate=0.001_batchSize=128_memorySize=5000_prioritizedMemory=True_lrs=True_smartReward=True_betaIncrement=Truedueling=True_double=True_gradientClipping=True_09:28.csv':
            running_avg = scores.rolling(window=window_size, min_periods=1).mean()
            if running_avg[100] > 0.53:
                all_scores.append(scores)
                plt.plot(running_avg, linewidth=0.8)

    # Calculate the mean of all scores
    mean_scores = np.mean(all_scores, axis=0)

    # Calculate the rolling mean of the mean scores
    mean_rolling_mean = pd.Series(mean_scores).rolling(window=100, min_periods=1).mean()

    # Plot the mean line
    plt.plot(mean_rolling_mean, linewidth=1, color='black', label='Mean')

    # dense grid
    plt.grid(True)
    plt.yticks(np.arange(0, 1.1, 0.1))

    # Adjust the legend with a smaller font size and position outside the plot
    plt.legend()

    plt.ylim(0, 1)
    plt.xlim(0, 1500)
    plt.show()


def plot_10_episode_policy_evaluation_all_files():
    """
    plot the 10 episode policy evaluation for all files in the directory.
    These are already averages of 10 episodes, so no need to do a running average.
    """
    directory = 'performances/10_episode_policy_evaluations/'

    all_scores = []

    for filename in os.listdir(directory):
        if filename == "testScore_11:32.csv":

            file_path = os.path.join(directory, filename)
            data = pd.read_csv(file_path)
            scores = data['test_score']
            all_scores.append(scores)
            plt.plot(scores, linewidth=0.5)

    # Calculate the mean of all scores
    mean_scores = np.mean(all_scores, axis=0)

    # Calculate the rolling mean of the mean scores
    mean_rolling_mean = pd.Series(mean_scores).rolling(window=10).mean()

    # Plot the mean line
    plt.plot(mean_rolling_mean, linewidth=2, color='black', label='Mean')

    # dense grid
    plt.grid(True)
    plt.yticks(np.arange(0, 1.1, 0.1))

    # Adjust the legend with a smaller font size and position outside the plot
    plt.legend()

    plt.ylim(0, 1)
    plt.show()


plot_running_average_all_files()
# plot_10_episode_policy_evaluation_all_files()


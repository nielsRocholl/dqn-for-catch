import os

import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import glob


def visualize(state: np.ndarray, enlarging_factor=5) -> None:
    """
    Visualize the state of the game using OpenCV
    :param state: state of the game
    :param enlarging_factor: factor to enlarge the state for better visualization
    :param delay: delay between visualizing frames in milliseconds
    :return: None
    """
    # Remove the batch dimension
    state = np.squeeze(state)
    size = 84 * enlarging_factor

    for i in range(state.shape[-1]):
        single_frame = state[..., i]
        resized_frame = cv2.resize(single_frame, (size, size))
        color_frame = np.zeros((size, size, 3), dtype=np.uint8)

        # Set purple color for the ball and paddle
        color_frame[resized_frame == 1] = [255, 230, 0]

        # Set orange color for the background
        color_frame[resized_frame == 0] = [75, 0, 88]

        cv2.imshow(f'Catch Game Frame {i + 1}', color_frame)
        cv2.waitKey(10)


def destroy() -> None:
    """
    Destroy all windows
    :return: None
    """
    cv2.destroyAllWindows()


def load_rewards_from_csv_files(path: str = "performance/"):
    rewards = []
    file_names = []

    for file in os.listdir(path):
        if file.endswith(".csv"):
            file_path = os.path.join(path, file)
            file_names.append(file[:-4])  # Remove .csv extension
            df = pd.read_csv(file_path)
            reward = df.iloc[:, 1].values
            rewards.append(reward)

    return rewards, file_names


def plot():
    def moving_average(data, window_size):
        return data.rolling(window=window_size).mean()

    def plot_data(files, title, moving_avg=False, window_size=10):
        plt.clf()
        for file in files:
            # get the second column of the csv file
            data = pd.read_csv(file).iloc[:, 1]
            label = ''.join(c for c in file.split('/')[-1] if c.isdigit() or c == '_')
            if moving_avg:
                data = moving_average(data, window_size)
            plt.plot(data, label=file[0:10])
        plt.title(title)
        plt.legend()
        plt.xlabel('Episodes')
        plt.show()

    reward_files = glob.glob('performances/reward*.csv')
    loss_files = glob.glob('performances/loss*.csv')
    wins_files = glob.glob('performances/wins*.csv')

    plot_data(reward_files, 'Reward Moving Average', moving_avg=True)
    plot_data(loss_files, 'Loss Moving Average', moving_avg=True)
    plot_data(wins_files, 'Wins vs Episodes')

plot()



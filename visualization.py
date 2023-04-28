import os

import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd


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


def plot_moving_average(window_size: int = 100, path: str = "performances/") -> None:
    rewards, file_names = load_rewards_from_csv_files(path)
    # increase figure dpi
    plt.figure(dpi=600)

    for r, name in zip(rewards, file_names):
        moving_average = []
        for i in range(len(r)):
            if i < window_size:
                window = r[:i + 1]
            else:
                window = r[i - window_size:i + 1]
            moving_average.append(np.sum(window) / len(window))
        # remove all letters from the file name, only keep nunbers
        plt.plot(moving_average, label=name)

    plt.ylabel(f"Rewards Moving Average (Window Size = {window_size})")
    plt.xlabel("Episodes")
    plt.legend()
    plt.show()

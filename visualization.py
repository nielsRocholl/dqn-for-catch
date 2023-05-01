import numpy as np
import cv2


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

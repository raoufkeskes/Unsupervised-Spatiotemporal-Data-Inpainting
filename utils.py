import torch.nn.functional as F
import torch


def read_video(path):
    """
    a Helper to read the full video from the path in batches of 35 or less(choose a number that divides the total
    number of frames). Means if the video contains 300 frames then this function returns a tensor of size
    (10, 3, 30, 64, 64)
    :param path:
    :return:
    """
    pass


def save_video(video, path):
    """
    Given a tensor of shape (N, C, T, H, W) represting a video and a path, save this video to this path.
    :param video:
    :param path:
    :return:
    """
    pass


if __name__ == "__main__":
    print("hello")
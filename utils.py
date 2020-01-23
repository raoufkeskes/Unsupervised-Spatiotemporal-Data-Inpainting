import torch.nn.functional as F
import torch
import cv2
import numpy as np


def read_video(path, size=(64, 64), inference=False):
    """
    a Helper to read the full video from the path in batches of 10 frames. Means if the video contains 300 frames
    then this function returns a tensor of size (30, 3, 10, 64, 64)
    :param inference:
    :param size:
    :param path:
    :return:
    """
    video = []
    batch = []
    cap = cv2.VideoCapture(path)
    while 1:
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame,size)[None]
            batch.append(frame)
            if len(batch) == 10:
                video.append(np.concatenate(batch)[None])
                batch = []
        else:
            break
    video = np.concatenate(video)
    cap.release()
    if inference:
        video = np.transpose(video, (0, 4, 1, 2, 3))
        video = video/255.
        video = torch.from_numpy(video)
    return video


def save_video(video, path, inference=True):
    """
    Given a numpy array of shape (N, C, T, H, W) represting a video and a path, save this video to this path.
    :param inference:
    :param video:
    :param path:
    :return:
    """
    if inference:
        video = (255*video).astype('uint8')
        video = np.transpose(video, (0, 2, 3, 4, 1))

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(path, fourcc, 20.0, video.shape[2:4])
    for batch in video:
        for frame in batch:
            out.write(frame)

    out.release()
    print("Saved!")


if __name__ == "__main__":

    path = 'VOT-Ball.mp4'
    x = read_video(path, inference=True)
    y = read_video(path)

    print(x.shape, x.min().item(), x.max().item())
    print(y.shape, y.min(), y.max())

    save_video(x.cpu().numpy(), 'test.avi')

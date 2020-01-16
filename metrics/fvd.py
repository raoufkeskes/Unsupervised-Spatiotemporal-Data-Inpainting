# Copyright 2019 The Google Research Authors.
# Original repo:
# https://github.com/google-research/google-research/blob/master/frechet_video_distance/frechet_video_distance.py
"""Minimal Reference implementation for the Frechet Video Distance (FVD).
FVD is a metric for the quality of video generation models. It is inspired by
the FID (Frechet Inception Distance) used for images, but uses a different
embedding to be better suitable for videos.
"""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import argparse

import six
import tensorflow as tf
import tensorflow_gan as tfgan
import tensorflow_hub as hub
import numpy as np
from glob import glob
import cv2
from tqdm import tqdm
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def preprocess(videos, target_resolution):
    """Runs some preprocessing on the videos for I3D model.
  Args:
    videos: <T>[batch_size, num_frames, height, width, depth] The videos to be
      preprocessed. We don't care about the specific dtype of the videos, it can
      be anything that tf.image.resize_bilinear accepts. Values are expected to
      be in the range 0-255.
    target_resolution: (width, height): target video resolution
  Returns:
    videos: <float32>[batch_size, num_frames, height, width, depth]
  """
    videos_shape = videos.shape.as_list()
    all_frames = tf.reshape(videos, [-1] + videos_shape[-3:])
    resized_videos = tf.image.resize_bilinear(all_frames, size=target_resolution)
    target_shape = [videos_shape[0], -1] + list(target_resolution) + [3]
    output_videos = tf.reshape(resized_videos, target_shape)
    scaled_videos = 2. * tf.cast(output_videos, tf.float32) / 255. - 1
    return scaled_videos


def _is_in_graph(tensor_name):
    """Checks whether a given tensor does exists in the graph."""
    try:
        tf.get_default_graph().get_tensor_by_name(tensor_name)
    except KeyError:
        return False
    return True


def create_id3_embedding(videos):
    """Embeds the given videos using the Inflated 3D Convolution network.
  Downloads the graph of the I3D from tf.hub and adds it to the graph on the
  first call.
  Args:
    videos: <float32>[batch_size, num_frames, height=224, width=224, depth=3].
      Expected range is [-1, 1].
  Returns:
    embedding: <float32>[batch_size, embedding_size]. embedding_size depends
               on the model used.
  Raises:
    ValueError: when a provided embedding_layer is not supported.
  """

    batch_size = 16
    module_spec = "https://tfhub.dev/deepmind/i3d-kinetics-400/1"

    # Making sure that we import the graph separately for
    # each different input video tensor.
    module_name = "fvd_kinetics-400_id3_module_" + six.ensure_str(
        videos.name).replace(":", "_")

    assert_ops = [
        tf.Assert(
            tf.reduce_max(videos) <= 1.001,
            ["max value in frame is > 1", videos]),
        tf.Assert(
            tf.reduce_min(videos) >= -1.001,
            ["min value in frame is < -1", videos]),
        tf.assert_equal(
            tf.shape(videos)[0],
            batch_size, ["invalid frame batch size: ",
                         tf.shape(videos)],
            summarize=6),
    ]
    with tf.control_dependencies(assert_ops):
        videos = tf.identity(videos)

    module_scope = "%s_apply_default/" % module_name

    # To check whether the module has already been loaded into the graph, we look
    # for a given tensor name. If this tensor name exists, we assume the function
    # has been called before and the graph was imported. Otherwise we import it.
    # Note: in theory, the tensor could exist, but have wrong shapes.
    # This will happen if create_id3_embedding is called with a frames_placehoder
    # of wrong size/batch size, because even though that will throw a tf.Assert
    # on graph-execution time, it will insert the tensor (with wrong shape) into
    # the graph. This is why we need the following assert.
    video_batch_size = int(videos.shape[0])
    assert video_batch_size in [batch_size, -1, None], "Invalid batch size"
    tensor_name = module_scope + "RGB/inception_i3d/Mean:0"
    if not _is_in_graph(tensor_name):
        i3d_model = hub.Module(module_spec, name=module_name)
        i3d_model(videos)

    # gets the kinetics-i3d-400-logits layer
    tensor_name = module_scope + "RGB/inception_i3d/Mean:0"
    tensor = tf.get_default_graph().get_tensor_by_name(tensor_name)
    return tensor


def calculate_fvd(real_activations, generated_activations):
    """Returns a list of ops that compute metrics as funcs of activations.
  Args:
    real_activations: <float32>[num_samples, embedding_size]
    generated_activations: <float32>[num_samples, embedding_size]
  Returns:
    A scalar that contains the requested FVD.
  """
    return tfgan.eval.frechet_classifier_distance_from_activations(
        real_activations, generated_activations)


def getVideos(path):
    """Returns a compact numpy array containing all the videos frames in the path given.
    :param
    path (string): full path the video dataset  in the form of:
                            path_gen/folder_name_1/frame1.ext  path_gen/folder_name_1/frame2.ext ...
                            path_gen/folder_name_2/frame1.ext  path_gen/folder_name_2/frame2.ext ...
                            path_gen/folder_name_3/frame1.ext  path_gen/folder_name_3/frame2.ext ...
    :returns
    videos : a numpy array containing all videos. The shape is [number_of_videos, frames, frame_height, frame_width, 3]
    the number_of_videos has to be multiple of 16 as indicated by the authors of the original repo
    """
    videos = []
    video_paths = glob(path + "/*/")
    video_paths = video_paths[:len(video_paths)//16 * 16]
    for vp in tqdm(video_paths):
        frames = []
        for frame_path in sorted(glob(vp+"/*")):
            frame = cv2.imread(frame_path)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame[None])
        videos.append(np.concatenate(frames)[None])
    videos = np.concatenate(videos)
    return videos


def getFVD(path_real, path_gen):
    real_videos = getVideos(path_real)
    gen_videos = getVideos(path_gen)
    with tf.Graph().as_default():
        real_videos = tf.convert_to_tensor(real_videos)
        gen_videos = tf.convert_to_tensor(gen_videos)
        result = calculate_fvd(
            create_id3_embedding(preprocess(real_videos, (224, 224))),
            create_id3_embedding(preprocess(gen_videos, (224, 224))))

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.tables_initializer())
            ress = sess.run(result)
            print("FVD is: %.2f." % ress)
    return ress

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_r', '-pr', type=str, metavar='DIR', help='path to real dataset')
    parser.add_argument('--path_g', '-pg', type=str, metavar='DIR', help='path to generated dataset')
    args = parser.parse_args()
    fvd_score = getFVD(args.path_r, args.path_g)
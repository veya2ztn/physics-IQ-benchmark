# Copyright 2024 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""This script processes videos to generate binary masks for ROI detection.

It reads videos from a specified input directory, applies Gaussian blurring and
thresholding to identify intensity changes, and then performs morphological
operations to clean up the binary mask.
The processed binary masks are saved as videos in the specified output
directory.
"""

import argparse
import os

import cv2
import numpy as np


def process_video(in_path, out_video_path, threshold_value=10):
  """Processes a video to generate a binary mask for ROI detection.

  Args:
    in_path: Path to the input video file.
    out_video_path: Path to save the generated binary mask video.
    threshold_value: Threshold value for binary segmentation.
  """
  try:
    cap = cv2.VideoCapture(in_path)
    if not cap.isOpened():
      print(f"Error: Could not open video file {in_path}")
      return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    input_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(
        out_video_path, fourcc, fps, (width, height), isColor=False
    )

    # Read the first frame and initialize the running average
    ret, prev_frame = cap.read()
    if not ret:
      print(f"Error: Could not read the first frame from {in_path}")
      return

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.GaussianBlur(prev_gray, (5, 5), 0)

    avg_frame = prev_gray.astype("float")

    # Write the first frame as a fully black frame
    black_frame = np.zeros((height, width), dtype=np.uint8)
    out.write(black_frame)
    generated_frame_count = 1

    frame_counter = 1

    # Define a kernel for morphological operations
    kernel = np.ones((5, 5), np.uint8)

    while True:
      ret, frame = cap.read()
      if not ret:
        break

      gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      gray_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)

      # Update the running average with a higher update rate
      # to reduce trailing
      cv2.accumulateWeighted(gray_frame, avg_frame, 0.3)
      avg_gray_frame = cv2.convertScaleAbs(avg_frame)

      # Calculate the absolute difference between the current frame
      # and the averaged frame
      frame_diff = cv2.absdiff(gray_frame, avg_gray_frame)

      # Create a binary frame based on intensity changes
      _, binary_frame = cv2.threshold(
          frame_diff, threshold_value, 255, cv2.THRESH_BINARY
      )

      # Apply morphological operations to clean up the binary frame
      binary_frame = cv2.morphologyEx(binary_frame, cv2.MORPH_OPEN, kernel)
      binary_frame = cv2.morphologyEx(binary_frame, cv2.MORPH_CLOSE, kernel)

      # Write binary frame to output video
      out.write(binary_frame)
      generated_frame_count += 1

      # Update frame counter
      frame_counter += 1

    cap.release()
    out.release()
    print(f"Processed video saved at {out_video_path}")
    print(f"Input video frame count: {input_frame_count}")
    print(f"Number of generated frames: {generated_frame_count}")

  except cv2.error as e:
    print(f"Error processing video {in_path}: {e}")


if __name__ == "__main__":
  parser = argparse.ArgumentParser(
      description="Process videos for binary segmentation."
  )
  parser.add_argument(
      "--input_parent_directory",
      type=str,
      required=True,
      help="Path to the parent directory containing input videos.",
  )
  parser.add_argument(
      "--output_parent_directory",
      type=str,
      required=True,
      help="Path to the parent directory for saving output videos.",
  )
  parser.add_argument(
      "--threshold_value",
      type=int,
      default=10,
      help="Threshold value for binary segmentation.",
  )

  args = parser.parse_args()

  if not os.path.exists(args.output_parent_directory):
    os.makedirs(args.output_parent_directory)

  # Assert that the directory now exists
  assert os.path.exists(
      args.output_parent_directory
  ), f"Failed to create the directory: {args.output_parent_directory}"

  for root, subdirs, _ in os.walk(args.input_parent_directory):
    for subdir in subdirs:
      input_directory = os.path.join(root, subdir)
      relative_path = os.path.relpath(
          input_directory, args.input_parent_directory
      )
      output_directory = os.path.join(
          args.output_parent_directory, relative_path
      )

      # Ensure the output subdirectory exists
      if not os.path.exists(output_directory):
        os.makedirs(output_directory)

      # Process each video file in the current subdirectory
      for filename in os.listdir(input_directory):
        if filename.endswith(".mp4"):
          input_path = os.path.join(input_directory, filename)
          output_video_path = os.path.join(output_directory, filename)
          print(
              f"Processing {filename} from {input_directory} to"
              f" {output_directory}..."
          )
          process_video(input_path, output_video_path, args.threshold_value)

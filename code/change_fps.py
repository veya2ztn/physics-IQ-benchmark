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

"""Change the FPS of videos in a CSV file.

This script takes a CSV file as input, which should contain a column with the
paths to the videos. It changes the FPS of these videos to a new value specified
by the user, and saves the new videos in a specified output directory.

The script also has an option to ensure that a specific frame (e.g., the frame
at
3 seconds) is the same in both the original and new videos. This is useful for
maintaining synchronization between videos.
"""

import argparse
import csv
import os

from moviepy.editor import ImageSequenceClip
from moviepy.editor import VideoFileClip
import numpy as np
from PIL import Image


# takes in a csv file with "video_path" being the name of file
def change_video_fps(csv_file, fps_new, output_column_name, input_dir,
                     output_dir):
  """Changes the FPS of videos in a CSV file.

  Args:
    csv_file: Path to the CSV file containing video paths.
    fps_new: New frames per second (FPS) value.
    output_column_name: Name of the column to store the new video path.
    input_dir: Directory containing the input videos.
    output_dir: Directory to save the output videos.
  """
  print("Starting FPS change process...")
  os.makedirs(output_dir, exist_ok=True)

  with open(csv_file, newline="") as csvfile:
    reader = csv.DictReader(csvfile)

    # Check if the output column exists, and if not, add it
    if isinstance(reader.fieldnames, list):
      if output_column_name not in reader.fieldnames:
        reader.fieldnames.append(output_column_name)

    updated_rows = []

    for row in reader:
      video_path_list = row["video_path"]
      video_path = os.path.join(input_dir, video_path_list)
      print("Processing:", video_path.split("/")[-1])
      if not os.path.exists(video_path):
        print(f"File not found: {video_path}")
        continue

      if video_path.split(".")[-1] != "mp4":
        print("Not an MP4 file, skipping.")
        continue

      try:
        # Load the video clip
        clip = VideoFileClip(video_path)
        fps_original = clip.fps
        duration = clip.duration

        print("Duration is:", duration)

        # Extract frames from the original video
        frames = [
            Image.fromarray(frame)
            for frame in clip.iter_frames(fps=fps_original, dtype="uint8")
        ]

        # Calculate the total number of new frames required
        frame_count_original = len(frames)
        frame_count_new = int(duration * fps_new)

        print(
            f"Original frames: {frame_count_original}, New frames:"
            f" {frame_count_new}"
        )

        # Ensure the frame at the end of third second (swith=ch point)
        # is the same since changing fps will change the number of frames
        original_3s_frame = int(
            3 * fps_original - 1
        )  # 90th frame in original video
        new_3s_frame = int(3 * fps_new - 1)  # Frame index in new video

        # Interpolate frames to match the new frame rate
        frames_new = []
        for j in range(frame_count_new):
          if j == new_3s_frame:
            # Ensure the frame at 3 seconds is the same as the 90th frame
            # in the original video
            frames_new.append(frames[original_3s_frame])
          else:
            # Regular interpolation process
            alpha = j * (frame_count_original - 1) / (frame_count_new - 1)
            idx = int(alpha)
            alpha -= idx

            # Get the two frames to interpolate between
            frame1 = frames[idx]
            frame2 = frames[min(idx + 1, frame_count_original - 1)]

            # Convert frames to numpy arrays for interpolation
            frame1_np = np.array(frame1).astype(np.float32)
            frame2_np = np.array(frame2).astype(np.float32)

            # Linear interpolation between two frames
            frame_interp_np = (1 - alpha) * frame1_np + alpha * frame2_np
            frame_interp = Image.fromarray(frame_interp_np.astype(np.uint8))

            frames_new.append(frame_interp)

        # Convert PIL images back to numpy arrays for moviepy
        frames_new_np = [np.array(frame) for frame in frames_new]
        new_clip = ImageSequenceClip(frames_new_np, fps=fps_new)

        # Create a new file name by adding the new FPS to the beginning
        new_file_name = f"{os.path.basename(video_path)}"
        new_file_path = os.path.join(output_dir, new_file_name)

        # Save the resampled video
        new_clip.write_videofile(new_file_path, codec="libx264")

        # Add or update the output column with the new video path
        row[output_column_name] = new_file_path
        updated_rows.append(row)

      except Exception as e:  # pylint: disable=broad-exception-caught
        print(f"Error processing {video_path}: {e}")

  print("FPS change process completed.")

if __name__ == "__main__":
  # Parse command-line arguments
  parser = argparse.ArgumentParser(
      description="Process video frames at a new frame rate."
  )
  parser.add_argument(
      "csv_file", type=str, help="Path to the CSV file containing video paths."
  )
  parser.add_argument(
      "fps_new", type=int, help="New frames per second (FPS) value."
  )
  parser.add_argument(
      "output_column_name",
      type=str,
      help="Name of the column to store the new video path.",
  )
  parser.add_argument(
      "input_dir", type=str, help="Directory containing the input videos."
  )
  parser.add_argument(
      "output_dir", type=str, help="Directory to save the output videos."
  )

  args = parser.parse_args()

  # Call the function with the parsed arguments
  change_video_fps(
      args.csv_file,
      args.fps_new,
      args.output_column_name,
      args.input_dir,
      args.output_dir,
  )

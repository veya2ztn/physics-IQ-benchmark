# Video Processing and Analysis Toolkit for Physics IQ project.

This is the code for running Physics-IQ to reproduce experiments from the paper.

## Installation

To install the required dependencies for this project, ensure you have Python 3 installed. Then, run the following command to install the necessary packages:

```bash
pip install -r requirements.txt
```

## Usage

### Generating Binary Mask Videos

Calculating metrics first requires generating binary mask videos of moving objects.
To generate binary mask videos from original videos, use the `generate_binary_mask_video.py` script. Here's how to do it:

```bash
python generate_binary_mask_video.py --input_parent_directory <input_directory> --output_parent_directory <output_directory> --threshold_value <threshold>
```

**Parameters:**
- `--input_parent_directory`: The path to the parent directory containing input videos (in `.mp4` format).
- `--output_parent_directory`: The path to the parent directory for saving the output binary mask videos.
- `--threshold_value`: The threshold value for binary segmentation - the higher the value, the less sensitive to movements the mask generation  (default is `10`).

**Example:**
```bash
python generate_binary_mask_video.py --input_parent_directory ./input_videos --output_parent_directory ./output_masks --threshold_value 10
```

### Calculating Metrics and Plotting

`change_fps.py` can be used to change the fps of input videos to the evaluation pipeline.
To calculate metrics from your videos and generate plots, utilize the `calculate_metrics_and_plot.py` script:

```bash
python calculate_metrics_and_plot.py <real_folder> <generated_folder> <binary_real_folder> <binary_generated_folder> <csv_file_path> <fps_list> <video_time_selection>
```

**Parameters:**
- `<real_folder>`: Path to the folder containing real videos.
- `<generated_folder>`: Path to the folder containing generated videos.
- `<binary_real_folder>`: Path to the folder containing binary masks for real videos.
- `<binary_generated_folder>`: Path to the folder containing binary masks for generated videos.
- `<csv_file_path>`: File path where the results will be saved as a CSV file.
- `<fps_list>`: List of frames per second (FPS) values for each video.
- `<video_time_selection>`: Specifies which part of the video to process (e.g., 'first', 'last').

**Example:**
```bash
python calculate_metrics_and_plot.py real_videos/ generated_videos/ binary_masks/ binary_generated_masks/ output_metrics.csv 30 first
```

## Citing this work

The citation details will be updated once the project has been published.

```latex
@article{publicationname,
      title={Publication Name},
      author={Author One and Author Two and Author Three},
      year={2024},
}
```

## License and disclaimer

Copyright 2024 DeepMind Technologies Limited

All software is licensed under the Apache License, Version 2.0 (Apache 2.0);
you may not use this file except in compliance with the Apache 2.0 license.
You may obtain a copy of the Apache 2.0 license at:
https://www.apache.org/licenses/LICENSE-2.0

All other materials are licensed under the Creative Commons Attribution 4.0
International License (CC-BY). You may obtain a copy of the CC-BY license at:
https://creativecommons.org/licenses/by/4.0/legalcode

Unless required by applicable law or agreed to in writing, all software and
materials distributed here under the Apache 2.0 or CC-BY licenses are
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
either express or implied. See the licenses for the specific language governing
permissions and limitations under those licenses.

This is not an official Google product.

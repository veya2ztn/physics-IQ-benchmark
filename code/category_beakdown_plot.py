# Copyright 2025 DeepMind Technologies Limited
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

"""Calculate metrics and plot results from physics-IQ benchmark videos."""

from calculate_iq_score import parse_list_of_floats

import argparse
from multiprocessing import pool
import os
import re

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tqdm


def ensure_length(lst, expected_length):
  if len(lst) < expected_length:
    return lst + [np.nan] * (expected_length - len(lst))
  return lst[:expected_length]


def get_average_view(df_columns, scenario_df):
  """Calculate the average value of multiple columns in a DataFrame for a specific scenario.

  Args:
    df_columns: A list of column names to average.
    scenario_df: A DataFrame containing the scenario data.

  Returns:
    A list representing the average value of the specified columns.
  """
  lists = []
  for column in df_columns:
    if column in scenario_df.columns:
      for value in scenario_df[column]:
        if isinstance(value, str):
          parsed_values = parse_list_of_floats(value)
          if parsed_values:
            lists.append(parsed_values)
  if lists:
    # The videos are 5 seconds long
    expected_length = scenario_df['fps'].iloc[0] * 5
    # print("expected length ", expected_length)
    lists = [ensure_length(lst, expected_length) for lst in lists]
    return np.mean(lists, axis=0).tolist()
  return []


def plot_metrics(
  csv_files, output_folder, fps_list, include_repeated_scenario=True
):
  """Plot metrics from CSV files.

  Args:
    csv_files: Paths to the CSV files containing scenarios and MSE values.
    output_folder: Path to the folder where plots will be saved.
    fps_list: List of FPS values for each CSV file.
    include_repeated_scenario: Whether to include the repeated scenario metrics in the plots.
  """
  # Define color map for models
  color_map = {
    'Lumiere_i2v': '#17becf',  # Light Blue
    'Lumiere_multiframe': '#1c69a7',  # Darker Blue
    'VideoPoet_i2v': '#ff7f0e',  # Brighter Orange
    'VideoPoet_multiframe': '#CC5500',  # Darker Reddish-Orange
    'SVD': '#9467bd',  # Purple
    'Pika': '#FFD700',  # golden yellow
    'Runway': '#2ca02c',  # green
    'Sora' : '#ff0606' #red
  }


  category_metric_folder = os.path.join(output_folder, 'category_breakdown')

  dataframes = {}
  for csv_name, frame_per_second in zip(csv_files, fps_list):
    model_name = os.path.splitext(os.path.basename(csv_name))[0]
    try:
      df = pd.read_csv(csv_name)
      df['fps'] = int(frame_per_second)  # Ensure FPS is an integer
      dataframes[model_name] = df
    except FileNotFoundError as e:
      print(f'Error reading {csv_name}: {e}')
      continue

  # Ensure there are valid dataframes to work with
  if not dataframes:
    raise ValueError('No valid CSV files provided.')

  first_model = next(iter(dataframes))
  if 'scenario' not in dataframes[first_model].columns:
    print(f"The 'scenario' column is missing in {first_model}.")
    return

  scenarios = dataframes[first_model]['scenario'].unique()

  # Generate average model metrics plots for each category
  category_data = {}
  for model_name, df in dataframes.items():
      for category in df['category'].unique():
          if category not in category_data:
              category_data[category] = {
                  'mse': {},
                  'iou': {},
                  'spatial_iou': {},
                  'spatiotemporal_iou': {},
                  'scenario_count': 0,  # Track scenario count for the category
              }

          # Filter scenarios by category
          category_scenarios = df[df['category'] == category]
          if category_scenarios.empty:
              continue

          # Set scenario count (only once for the category)
          if category_data[category]['scenario_count'] == 0:
              scenario_count = len(category_scenarios)
              category_data[category]['scenario_count'] = scenario_count
              print(f"Category: {category}, Scenario Count: {scenario_count}")  # Print scenario count for verification

          # Calculate averages for metrics
          v1_mse = get_average_view(
              category_scenarios[['v1_mse_perspective-left', 'v1_mse_perspective-center', 'v1_mse_perspective-right']],
              category_scenarios,
          )
          v2_mse = get_average_view(
              category_scenarios[['v2_mse_perspective-left', 'v2_mse_perspective-center', 'v2_mse_perspective-right']],
              category_scenarios,
          ) if include_repeated_scenario else []
          iou_v1 = get_average_view(
              category_scenarios[['iou_v1_perspective-left', 'iou_v1_perspective-center', 'iou_v1_perspective-right']],
              category_scenarios,
          )
          iou_v2 = get_average_view(
              category_scenarios[['iou_v2_perspective-left', 'iou_v2_perspective-center', 'iou_v2_perspective-right']],
              category_scenarios,
          ) if include_repeated_scenario else []

          # Ensure scalars
          v1_mse = np.mean(v1_mse) if isinstance(v1_mse, (list, np.ndarray)) else v1_mse
          v2_mse = np.mean(v2_mse) if isinstance(v2_mse, (list, np.ndarray)) else v2_mse
          iou_v1 = np.mean(iou_v1) if isinstance(iou_v1, (list, np.ndarray)) else iou_v1
          iou_v2 = np.mean(iou_v2) if isinstance(iou_v2, (list, np.ndarray)) else iou_v2

          # Check spatial and spatiotemporal IOU columns
          if 'spatial_iou_v1' in category_scenarios.columns:
              category_data[category]['spatial_iou'][model_name] = category_scenarios[
                  'spatial_iou_v1'
              ].mean()
          if 'spatiotemporal_iou_v1' in category_scenarios.columns:
              category_data[category]['spatiotemporal_iou'][model_name] = (
                  category_scenarios['spatiotemporal_iou_v1'].mean()
              )

          # Store metrics
          category_data[category]['mse'][model_name] = (
              (v1_mse + v2_mse) / 2 if include_repeated_scenario else v1_mse
          )
          category_data[category]['iou'][model_name] = (
              (iou_v1 + iou_v2) / 2 if include_repeated_scenario else iou_v1
          )

  # Create category-specific plots
  for category, metrics in category_data.items():
    category_folder = os.path.join(category_metric_folder, category)
    os.makedirs(category_folder, exist_ok=True)

    for metric_key in ['mse', 'iou', 'spatial_iou', 'spatiotemporal_iou']:
        plt.figure(figsize=(10, 6))
        model_names = list(metrics[metric_key].keys())
        values = list(metrics[metric_key].values())

        # Sort values
        sorted_indices = np.argsort(values) if metric_key == 'mse' else np.argsort(values)[::-1]
        model_names = [model_names[i] for i in sorted_indices]
        values = [values[i] for i in sorted_indices]

        # Calculate physical variance
        category_df = dataframes[first_model][dataframes[first_model]['category'] == category]
        physical_variance = None
        if metric_key == 'mse':
            physical_variance = np.mean([
                category_df[f'variance_mse_{view}'].apply(parse_list_of_floats).explode().mean()
                for view in ['perspective-left', 'perspective-center', 'perspective-right'] if f'variance_mse_{view}' in category_df.columns
            ])
        elif metric_key == 'iou':
            physical_variance = np.mean([
                category_df[f'variance_iou_{view}'].apply(parse_list_of_floats).explode().mean()
                for view in ['perspective-left', 'perspective-center', 'perspective-right'] if f'variance_iou_{view}' in category_df.columns
            ])
        elif metric_key == 'spatial_iou':
            physical_variance = np.mean([
                category_df[f'variance_spatial_{view}'].mean()
                for view in ['perspective-left', 'perspective-center', 'perspective-right'] if f'variance_spatial_{view}' in category_df.columns
            ])
        elif metric_key == 'spatiotemporal_iou':
            physical_variance = np.mean([
                category_df[f'variance_spatiotemporal_{view}'].mean()
                for view in ['perspective-left', 'perspective-center', 'perspective-right'] if f'variance_spatiotemporal_{view}' in category_df.columns
            ])

        # Bar plot
        colors = [color_map.get(name, '#333333') for name in model_names]
        plt.bar(model_names, values, color=colors)

        # Physical variance line
        if physical_variance is not None:
          plt.axhline(
              y=physical_variance, 
              color='black',        # Set the color to black
              linestyle='--',       # Dashed line style
              linewidth=2           # Increase line width for boldness
          )

        # Hide x-axis labels and titles
        plt.xticks([])
        plt.xlabel("")
        plt.ylabel("")
        plt.title("")  # Ensure the title is removed

        # Remove the right and top axis lines
        ax = plt.gca()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        # Hide the legend
        #plt.legend().set_visible(False)

        # Save and close the plot
        plt.tight_layout()
        plot_file = os.path.join(category_folder, f"{metric_key}_{category}.png")
        plt.savefig(plot_file, dpi=500)
        plt.close()


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Process videos and plot MSE.')
  parser.add_argument(
      '--csv_files',
      type=str,
      nargs='+',
      help='Paths to the CSV files containing scenarios and MSE values.',
  )
  parser.add_argument(
      '--fps',
      type=float,
      nargs='+',
      required=True,
      help='FPS for each CSV file. Must match the number of CSV files.',
  )

  args = parser.parse_args()

  plot_metrics(args.csv_files, './results', args.fps)

  # Exemplary usage:
  # python3 code/category_beakdown_plot.py --csv_files results/VideoPoet_multiframe.csv results/VideoPoet_i2v.csv results/Lumiere_multiframe.csv results/Lumiere_i2v.csv results/SVD.csv results/Runway.csv results/Pika.csv results/Sora.csv --fps 8 8 16 16 8 24 24 30


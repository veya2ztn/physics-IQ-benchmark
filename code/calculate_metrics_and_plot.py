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

"""Calculate metrics and plot results from physics-IQ benchmark videos."""

import argparse
from multiprocessing import pool
import os
import re
from typing import Any, Sequence
import warnings

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tqdm


def get_video_frame_count(filepath: str) -> int:
  """Get the total number of frames in a video."""
  if not os.path.exists(filepath):
    return 0
  cap = cv2.VideoCapture(filepath)
  total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
  cap.release()
  return total_frames


plt.rcParams.update({'axes.grid': False})  # Disable grid on plots


def parse_list_of_floats(
    value: str | Sequence[float] | Sequence[int]
) -> list[float]:
  """Parse a string representing a list of floats.

  This function takes a string input that represents a list of float numbers,
  separated by commas or spaces, and returns a list of floats.

  Args:
      value (str or Sequence[float] or Sequence[int]): If str, a string
                   representation of a list of floats.
                   The string can contain numbers separated by commas or spaces.
                   If list, a list of floats or ints.

  Returns:
      List[float]: A list of floats parsed from the input string. If the input
                   string is empty or invalid, an empty list is returned.
  """
  try:
    if isinstance(value, str) and (
        value.startswith('[') and value.endswith(']')
    ):
      # Use regular expression to identify numbers in the string, including
      # in scientific notation.
      return [
          float(x)
          for x in re.findall(
              r'[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?', value
          )
      ]
    elif isinstance(value, list):
      list_of_floats = []
      for x in value:
        if isinstance(x, (int, float)):
          list_of_floats.append(float(x))
        else:
          raise ValueError(f'Unsupported type: {type(x)} of value {x}')
      return list_of_floats

    else:
      return []
  except (ValueError, TypeError):
    return []


def ensure_length(lst: Sequence[float], expected_length: int) -> list[float]:
  if len(lst) < expected_length:
    return lst + [np.nan] * (expected_length - len(lst))
  return lst[:expected_length]


def get_average_view(
    df_columns: Sequence[str], scenario_df: pd.DataFrame
) -> Sequence[float]:
  """Calculate the average value of multiple columns in a df for a scenario.

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
    else:
      warnings.warn(f'Column {column} not found in scenario_df.')
  if lists:
    # The videos are 5 seconds long
    expected_length = scenario_df['fps'].iloc[0] * 5
    lists = [ensure_length(lst, expected_length) for lst in lists]
    return np.mean(lists, axis=0).tolist()
  return []


def create_bar_plot(
    model_names: Sequence[str],
    values: Sequence[float],
    metric_key: str,
    metric_label: str,
    physical_variance: float,
    folder: str,
) -> None:
  """Creates a bar plot for a given metric."""
    # Create bar plots for each model
  plt.bar(model_names, values, color=plt.cm.tab20.colors)
  plt.axhline(
      y=physical_variance,
      color='lightgray',
      linestyle='--',
      label='Physical Variance',
  )
  plt.xticks(rotation=45, ha='right')
  plt.ylabel(
      f'{metric_label} Value (higher=better)'
      if 'iou' in metric_key
      else f'{metric_label} Value (lower=better)'
  )
  plt.title(f'Average {metric_label} Across All Scenarios for Each Model')
  plt.grid(False)
  plt.legend()
  plt.tight_layout()
  plot_file = os.path.join(folder, f'average_{metric_key}.pdf')
  plt.savefig(plot_file)
  plt.close()


def plot_metrics(
    csv_files: Sequence[str],
    output_folder: str,
    fps_list: Sequence[float],
    include_repeated_scenario: bool = True,
) -> None:
  """Plot metrics from CSV files.

  Args:
    csv_files: Paths to the CSV files containing scenarios and MSE values.
    output_folder: Path to the folder where plots will be saved.
    fps_list: List of FPS values for each CSV file.
    include_repeated_scenario: Whether to include the repeated scenario metrics
      in the plots.
  """
  # Create output folders if they do not exist
  scenario_metric_folder = os.path.join(output_folder, 'scenario_metric')
  model_metric_folder = os.path.join(output_folder, 'model_metric')
  category_metric_folder = os.path.join(output_folder, 'category_metric')
  # Create specific subfolders for each metric
  mse_folder = os.path.join(scenario_metric_folder, 'mse')
  iou_folder = os.path.join(scenario_metric_folder, 'iou')
  spatial_iou_folder = os.path.join(scenario_metric_folder, 'spatial_iou')
  spatiotemporal_iou_folder = os.path.join(
      scenario_metric_folder, 'spatiotemporal_iou'
  )

  for folder in [
      mse_folder,
      iou_folder,
      spatial_iou_folder,
      spatiotemporal_iou_folder,
  ]:
    os.makedirs(folder, exist_ok=True)

  # Read CSV files into DataFrames
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

  # Iterate through scenarios to collect and plot data
  for scenario in scenarios:
    scenario_data = {}
    for model_name, df in dataframes.items():
      scenario_df = df[df['scenario'] == scenario]

      if scenario_df.empty:
        print(f'No valid data for scenario {scenario}. Skipping plot.')
        continue
      try:

        v1_mse = get_average_view(
            ['v1_mse_A', 'v1_mse_B', 'v1_mse_C'], scenario_df
        )
        if include_repeated_scenario and 'v2_mse_A' in scenario_df.columns:
          v2_mse = get_average_view(
              ['v2_mse_A', 'v2_mse_B', 'v2_mse_C'], scenario_df
          )
        else:
          v2_mse = []
        if 'iou_v1_A' in scenario_df.columns:
          iou_v1 = get_average_view(
              ['iou_v1_A', 'iou_v1_B', 'iou_v1_C'], scenario_df
          )
        else:
          iou_v1 = []
        if include_repeated_scenario and 'iou_v2_A' in scenario_df.columns:
          iou_v2 = get_average_view(
              ['iou_v2_A', 'iou_v2_B', 'iou_v2_C'], scenario_df
          )
        else:
          iou_v2 = []

        # Scalars for spatial and spatiotemporal IOU
        spatial_iou = scenario_df.iloc[0].get('spatial_iou_v1')
        spatiotemporal_iou = scenario_df.iloc[0].get('spatiotemporal_iou_v1')

        # Calculate physical variance for MSE
        physical_variance_mse = np.concatenate([
            scenario_df[f'variance_mse_{view}']
            .apply(parse_list_of_floats)
            .explode()
            .astype(float)
            for view in ['A', 'B', 'C']
            if f'variance_mse_{view}' in scenario_df.columns
        ]).tolist()

        upper_bound = get_average_view(
            ['upper_bound_A', 'upper_bound_B', 'upper_bound_C'], scenario_df
        )

        # Averaging v1 and v2 metrics if include_repeated_scenario is True
        mse = (
            np.mean([val for val in [v1_mse, v2_mse] if val], axis=0).tolist()
            if include_repeated_scenario
            else v1_mse
        )

        iou = (
            np.mean([val for val in [iou_v1, iou_v2] if val], axis=0).tolist()
            if include_repeated_scenario
            else iou_v1
        )

        scenario_data[model_name] = {
            'mse': mse,
            'iou': iou,
            'spatial_iou': spatial_iou,
            'spatiotemporal_iou': spatiotemporal_iou,
            'physical_variance_mse': physical_variance_mse,
            'upper_bound': upper_bound,
        }

      except (IndexError, TypeError, ValueError) as e:
        print(
            f'Error processing scenario {scenario} for model'
            f' {model_name}: {e}'
        )
        continue

    # Plotting metrics for the current scenario
    if not scenario_data:
      print(f'No valid data for scenario {scenario}. Skipping plot.')
      continue

    metrics_to_plot = {
        'mse': ('MSE Full Frame', mse_folder),
        'iou': ('IOU', iou_folder),
        'spatial_iou': ('Spatial IOU', spatial_iou_folder),
        'spatiotemporal_iou': ('Spatiotemporal IOU', spatiotemporal_iou_folder),
    }

    for metric_key, (metric_label, folder) in metrics_to_plot.items():
      plt.figure(figsize=(10, 6))

      if metric_key == 'mse' or metric_key == 'iou':
        for model_name, data in scenario_data.items():
          metric_values = data.get(metric_key, [])
          if metric_values:  # Ensure it's a list
            if isinstance(metric_values, (int, float)):
              metric_values = [metric_values]
            time_axis = np.linspace(
                0, 5, len(metric_values)
            )  # Normalize to 5 seconds
            plt.plot(
                time_axis, metric_values, label=f'{model_name}', linestyle='-'
            )
            plt.scatter(time_axis, metric_values, s=10)

        # Plot physical variance for MSE
        if metric_key == 'mse':
          if physical_variance_mse:  # Check if it's not empty
            plt.plot(
                np.linspace(0, 5, len(physical_variance_mse)),
                physical_variance_mse,
                color='lightgray',
                linestyle='--',
                label='Physical Variance',
            )

        elif metric_key in ['iou', 'spatial_iou', 'spatiotemporal_iou']:
          physical_variance_iou = np.mean([
              dataframes[first_model][f'variance_{metric_key}_{view}']
              .apply(parse_list_of_floats)
              .explode()
              .astype(float)
              .mean()
              for view in ['A', 'B', 'C']
              if f'variance_{metric_key}_{view}'
              in dataframes[first_model].columns
          ])
          plt.axhline(
              y=physical_variance_iou,
              color='lightgray',
              linestyle='--',
              label='Physical Variance',
          )

        plt.xlabel('Time (seconds)')
        plt.ylabel(
            f'{metric_label} Value (lower=better)'
            if metric_key == 'mse'
            else f'{metric_label} Value (higher=better)'
        )
        plt.title(f'{metric_label} for Scenario {scenario}')
        plt.legend()
        plt.grid(False)
        plot_file = os.path.join(folder, f'{scenario}_{metric_key}.pdf')
        plt.savefig(plot_file)
        plt.close()

      elif metric_key in ['spatial_iou', 'spatiotemporal_iou']:
        model_names = []
        values = []
        for model_name, data in scenario_data.items():
          value = data.get(metric_key)
          if value is not None:
            model_names.append(model_name)
            values.append(value)

        # Filter out None values to avoid empty plots
        if not values:
          print(f'No valid values for {metric_key}. Skipping plot.')
          continue

        # Sort model names and values based on metric key
        sorted_indices = np.argsort(values)[
            ::-1
        ]  # Sort IOU metrics in descending order
        model_names = [model_names[i] for i in sorted_indices]
        values = [values[i] for i in sorted_indices]

        plt.bar(model_names, values, label=metric_label)
        plt.xticks(rotation=45, ha='right')
        plt.ylabel(f'{metric_label} Value (higher=better)')
        plt.title(f'{metric_label} for Scenario {scenario}')
        plt.grid(False)
        plt.tight_layout()
        plot_file = os.path.join(folder, f'{scenario}_{metric_key}.pdf')
        plt.savefig(plot_file)
        plt.close()

  # Plotting metrics averaged over all scenarios for each model
  model_data = {}
  for model_name, df in dataframes.items():
    avg_data = {}
    for metric in [
        'v1_mse_A',
        'v1_mse_B',
        'v1_mse_C',
        'v2_mse_A',
        'v2_mse_B',
        'v2_mse_C',
        'iou_v1_A',
        'iou_v1_B',
        'iou_v1_C',
        'iou_v2_A',
        'iou_v2_B',
        'iou_v2_C',
        'variance_mse_A',
        'variance_mse_B',
        'variance_mse_C',
        'variance_iou_A',
        'variance_iou_B',
        'variance_iou_C',
        'variance_spatial_A',
        'variance_spatial_B',
        'variance_spatial_C',
        'variance_spatiotemporal_A',
        'variance_spatiotemporal_B',
        'variance_spatiotemporal_C',
    ]:
      if metric in df.columns:
        avg_data[metric] = (
            df[metric]
            .apply(parse_list_of_floats)
            .dropna()
            .apply(np.mean)
            .tolist()
        )

    if include_repeated_scenario:
      spatial_iou_values = []
      if 'spatial_iou_v1' in df.columns:
        spatial_iou_values.append(df['spatial_iou_v1'].mean())
      if 'spatial_iou_v2' in df.columns:
        spatial_iou_values.append(df['spatial_iou_v2'].mean())

      spatiotemporal_iou_values = []
      if 'spatiotemporal_iou_v1' in df.columns:
        spatiotemporal_iou_values.append(df['spatiotemporal_iou_v1'].mean())
      if 'spatiotemporal_iou_v2' in df.columns:
        spatiotemporal_iou_values.append(df['spatiotemporal_iou_v2'].mean())

      avg_data['spatial_iou'] = (
          np.mean(spatial_iou_values) if spatial_iou_values else None
      )
      avg_data['spatiotemporal_iou'] = (
          np.mean(spatiotemporal_iou_values)
          if spatiotemporal_iou_values
          else None
      )
    else:
      if 'spatial_iou_v1' in df.columns:
        avg_data['spatial_iou'] = df['spatial_iou_v1'].mean()
      if 'spatiotemporal_iou_v1' in df.columns:
        avg_data['spatiotemporal_iou'] = df['spatiotemporal_iou_v1'].mean()

    model_data[model_name] = avg_data

  # Generate average model metrics plots
  metrics_to_plot = {
      'mse': ('MSE Full Frame', model_metric_folder),
      'iou': ('IOU', model_metric_folder),
      'spatial_iou': ('Spatial IOU', model_metric_folder),
      'spatiotemporal_iou': ('Spatiotemporal IOU', model_metric_folder),
  }

  for metric_key, (metric_label, folder) in metrics_to_plot.items():
    plt.figure(figsize=(10, 6))
    names_and_values = []  # list storing tuples of (model_name, value)

    for model_name, data in model_data.items():
      if metric_key not in ['spatial_iou', 'spatiotemporal_iou']:
        metric_values = []
        for key in data:
          if metric_key in key:
            value = data[key]
            if isinstance(value, list):
              metric_values.extend(value)
            else:
              metric_values.append(value)

        if metric_values:
          avg_metric_value = np.mean(
              [val for val in metric_values if val is not None]
          )  # Average
          names_and_values.append((model_name, avg_metric_value))
      else:
        value = data.get(metric_key)
        names_and_values.append((model_name, value))

    # Filter out None values to avoid empty plots
    names_and_values = [(m, v) for m, v in names_and_values if v is not None]

    if not names_and_values:
      print(f'No valid values for {metric_key}. Skipping plot.')
      continue

    # Sort model names and values for metrics according to values.
    # Ascending order for MSE; descending order for IOU metrics.
    names_and_values = sorted(names_and_values,
                              key=lambda x: x[1],
                              reverse=metric_key != 'mse')

    # Adding physical variance
    first_model = next(iter(model_data.keys()))

    if metric_key == 'mse':
      physical_variance = np.mean([
          dataframes[first_model][f'variance_mse_{view}']
          .apply(parse_list_of_floats)
          .explode()
          .astype(float)
          .mean()  # Average across frames for each view
          for view in ['A', 'B', 'C']
          if f'variance_mse_{view}' in dataframes[first_model].columns
      ])
    elif metric_key in ['spatial_iou']:
      physical_variance = np.mean([
          dataframes[first_model][
              f'variance_spatial_{view}'
          ].mean()  # Directly average the single scalar values
          for view in ['A', 'B', 'C']
          if f'variance_spatial_{view}' in dataframes[first_model].columns
      ])
    elif metric_key in ['spatiotemporal_iou']:
      physical_variance = np.mean([
          dataframes[first_model][
              f'variance_spatiotemporal_{view}'
          ].mean()  # Directly average the single scalar values
          for view in ['A', 'B', 'C']
          if f'variance_spatiotemporal_{view}'
          in dataframes[first_model].columns
      ])
    else:
      raise ValueError(f'Unsupported metric key: {metric_key}')
    create_bar_plot(model_names=[m for m, _ in names_and_values],
                    values=[v for _, v in names_and_values],
                    metric_key=metric_key,
                    metric_label=metric_label,
                    physical_variance=physical_variance,
                    folder=folder)

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

      # Calculate the scenario count (you can use any model,
      # here we use the current one)
      if category_data[category]['scenario_count'] == 0:
        # Only set once for the category
        category_data[category]['scenario_count'] = len(category_scenarios)

      # Calculate averages for metrics in this category per model
      v1_mse = get_average_view(
          category_scenarios[['v1_mse_A', 'v1_mse_B', 'v1_mse_C']],
          category_scenarios,
      )
      if include_repeated_scenario:
        v2_mse = get_average_view(
            category_scenarios[['v2_mse_A', 'v2_mse_B', 'v2_mse_C']],
            category_scenarios,
        )
      else:
        v2_mse = []
      iou_v1 = get_average_view(
          category_scenarios[['iou_v1_A', 'iou_v1_B', 'iou_v1_C']],
          category_scenarios,
      )
      if include_repeated_scenario:
        iou_v2 = get_average_view(
            category_scenarios[['iou_v2_A', 'iou_v2_B', 'iou_v2_C']],
            category_scenarios,
        )
      else:
        iou_v2 = []

      # Ensure v1_mse and v2_mse are scalars
      v1_mse = (
          np.mean(v1_mse) if isinstance(v1_mse, (list, np.ndarray)) else v1_mse
      )
      v2_mse = (
          np.mean(v2_mse) if isinstance(v2_mse, (list, np.ndarray)) else v2_mse
      )

      # Ensure iou_v1 and iou_v2 are scalars
      iou_v1 = (
          np.mean(iou_v1) if isinstance(iou_v1, (list, np.ndarray)) else iou_v1
      )
      iou_v2 = (
          np.mean(iou_v2) if isinstance(iou_v2, (list, np.ndarray)) else iou_v2
      )

      # Check if the spatial IOU columns exist before accessing them
      if 'spatial_iou_v1' in category_scenarios.columns:
        category_data[category]['spatial_iou'][model_name] = category_scenarios[
            'spatial_iou_v1'
        ].mean()
      else:
        print(
            f"Warning: 'spatial_iou_v1' not found for category '{category}' and"
            f" model '{model_name}'."
        )

      if 'spatiotemporal_iou_v1' in category_scenarios.columns:
        category_data[category]['spatiotemporal_iou'][model_name] = (
            category_scenarios['spatiotemporal_iou_v1'].mean()
        )
      else:
        print(
            "Warning: 'spatiotemporal_iou_v1' not found for category"
            f" '{category}' and model '{model_name}'."
        )

      # Store metrics per model for each category
      category_data[category]['mse'][model_name] = (
          (v1_mse + v2_mse) / 2 if include_repeated_scenario else v1_mse
      )
      category_data[category]['iou'][model_name] = (
          (iou_v1 + iou_v2) / 2 if include_repeated_scenario else iou_v1
      )

  # Create category-specific subfolders and plots
  for category, metrics in category_data.items():
    category_folder = os.path.join(category_metric_folder, category)
    os.makedirs(category_folder, exist_ok=True)

    for metric_key in ['mse', 'iou', 'spatial_iou', 'spatiotemporal_iou']:
      plt.figure(figsize=(10, 6))
      model_names = list(metrics[metric_key].keys())
      values = list(metrics[metric_key].values())

      # Sorting based on metric key
      if metric_key == 'mse':
        sorted_indices = np.argsort(values)  # Ascending order
      else:
        sorted_indices = np.argsort(values)[::-1]  # Descending order

      model_names = [model_names[i] for i in sorted_indices]
      values = [values[i] for i in sorted_indices]

      # Adding physical variance to the plot
      first_model = next(iter(dataframes.keys()))
      if metric_key == 'mse':
        physical_variance = np.mean([
            dataframes[first_model][f'variance_mse_{view}']
            .apply(parse_list_of_floats)
            .explode()
            .astype(float)
            .mean()
            for view in ['A', 'B', 'C']
            if f'variance_mse_{view}' in dataframes[first_model].columns
        ])
      elif metric_key in ['spatial_iou']:
        physical_variance = np.mean([
            dataframes[first_model][
                f'variance_spatial_{view}'
            ].mean()  # Directly average the single scalar values
            for view in ['A', 'B', 'C']
            if f'variance_spatial_{view}' in dataframes[first_model].columns
        ])
      elif metric_key in ['spatiotemporal_iou']:
        physical_variance = np.mean([
            dataframes[first_model][
                f'variance_spatiotemporal_{view}'
            ].mean()  # Directly average the single scalar values
            for view in ['A', 'B', 'C']
            if f'variance_spatiotemporal_{view}'
            in dataframes[first_model].columns
        ])
      plot_barplots(model_names, values, metrics, metric_key,
                    physical_variance, category, category_folder)

  print(f'Plots have been saved to {output_folder}')


def plot_barplots(
    model_names: Sequence[str],
    values: Sequence[float],
    metrics: dict[str, Any],
    metric_key: str,
    physical_variance: float,
    category: str,
    category_folder: str) -> None:
  """Create bar plots for each model."""
  plt.bar(model_names, values, color=plt.cm.tab20.colors)
  plt.axhline(
      y=physical_variance,
      color='lightgray',
      linestyle='--',
      label='Physical Variance',
  )
  plt.xticks(rotation=45, ha='right')
  plt.ylabel(
      f'{metric_key.capitalize()} Value (higher=better)'
      if 'iou' in metric_key
      else f'{metric_key.capitalize()} Value (lower=better)'
  )
  plt.title(
      f'Average {metric_key.capitalize()} for Category'
      f' {category} (Scenarios: {metrics["scenario_count"]})'
  )
  plt.grid(False)
  plt.tight_layout()
  plot_file = os.path.join(category_folder, f'{category}_{metric_key}.pdf')
  plt.savefig(plot_file)
  plt.close()


def process_videos(
    real_folders: Sequence[str],
    generated_folders: Sequence[str],
    binary_real_folders: Sequence[str],
    binary_generated_folders: Sequence[str],
    csv_file_path: str,
    fps_list: Sequence[int],
    video_time_selection: str,
) -> None:
  """Goes through the videos and masks, and calculates metrics.

  This function processes a set of real and generated videos along with their
  corresponding binary masks. It calculates various metrics such as MSE,
  IOU, and others, and saves the results to a specified CSV file.

  Args:
      real_folders (list of str): A list of paths to folders containing real
        videos.
      generated_folders (list of str): A list of paths to folders containing
        generated videos.
      binary_real_folders (list of str): A list of paths to folders containing
        binary masks for real videos.
      binary_generated_folders (list of str): A list of paths to folders
        containing binary masks for generated videos.
      csv_file_path (str): The file path where the results will be saved as a
        CSV file.
      fps_list (list of int): A list of frames per second (FPS) values for each
        video.
      video_time_selection (str): Specifies which part of the video to process
        (e.g., 'first', 'last').

  Returns:
      None: This function does not return any value but saves the results to a
      CSV file.
  """

  def spatial_binary_masks(mask_frames: Sequence[np.ndarray]) -> np.ndarray:
    """Collapse the time dimension of binary masks into a single frame."""
    if not mask_frames:
      print(
          'Warning: Received empty mask_frames for spatial binary mask '
          'calculation.'
      )
      return np.zeros(
          (1, 1), dtype=np.uint8
      )  # Return an empty mask to prevent errors
    spatial_mask = np.max(mask_frames, axis=0)
    return (spatial_mask > 0).astype(np.uint8) * 255

  def spatiotemporal_binary_mask(mask_frames):
    """Calculate an spatiotemporal presence map by counting the number of frames with activity."""
    spatiotemporal_mask = np.sum(mask_frames, axis=0)
    # Keep the count of active frames as pixel values
    return spatiotemporal_mask.astype(np.uint8)

  scenario_data = []
  processed_scenarios = set()
  gen_video_duration_frames = get_video_frame_count(
      os.path.join(generated_folders, os.listdir(generated_folders)[0])
  )

  def mse_per_frame(video1, video2):
    """Calculate MSE per frame for two videos.

    Args:
      video1:
      video2:

    Returns:

    """
    frame_mses = []
    for frame1, frame2 in zip(video1, video2):
      mse = np.mean((frame1.flatten() - frame2.flatten()) ** 2)
      frame_mses.append(mse)
    return frame_mses

  def load_and_resize_video(
      filepath: str,
      start_frame: int,
      end_frame: int,
      target_size: tuple[int, int] = None,
      normalize: bool = True,
  ) -> list[np.ndarray]:
    """Load and resize a video.

    Args:
        filepath: Path to the video file.
        start_frame: Index of the first frame to load.
        end_frame: Index of the last frame to load.
        target_size: Desired size of the frames (width, height).
        normalize: Whether to normalize the pixel values to the range [0, 1].

    Returns:
        A list of frames from the video.
    """
    if not os.path.exists(filepath):
      print(f'File not found: {filepath}')
      return []
    cap = cv2.VideoCapture(filepath)
    frames = []
    frame_idx = 0
    while cap.isOpened():
      ret, frame = cap.read()
      if not ret:
        break
      if start_frame <= frame_idx < end_frame:
        if target_size:
          frame = cv2.resize(frame, target_size)
        if normalize:
          frame = frame / 255.0
        frames.append(frame)
      frame_idx += 1
    cap.release()
    return frames

  def iou_per_frame(mask1, mask2):
    """Calculate Intersection over Union (IoU) per frame for two binary masks."""
    iou_values = []
    for m1, m2 in zip(mask1, mask2):
      intersection = np.logical_and(m1, m2).sum()
      union = np.logical_or(m1, m2).sum()
      iou = intersection / union if union != 0 else 0
      iou_values.append(iou)
    return iou_values

  scenario_data = []
  processed_scenarios = set()
  gen_video_duration_frames = get_video_frame_count(
      os.path.join(generated_folders, os.listdir(generated_folders)[0])
  )

  consider_frames = fps_list * 5
  if video_time_selection == 'first':
    start_frame = 0
    end_frame = consider_frames
  else:
    start_frame = gen_video_duration_frames - (5 * fps_list)
    end_frame = gen_video_duration_frames

  progress_bar = tqdm.tqdm(
      total=len(
          set(
              '_'.join(f.rsplit('_', 2)[:-2])
              for f in os.listdir(real_folders)
              if f.endswith('.mp4')
          )
      ),
      desc='Processing scenarios',
  )
  if not os.path.exists(real_folders):
    print(f'Folder not found: {real_folders}')
    return

  spatial_maps_folder = os.path.join(os.getcwd(), 'spatial_maps')
  spatiotemporal_maps_folder = os.path.join(os.getcwd(), 'spatiotemporal_maps')
  os.makedirs(spatial_maps_folder, exist_ok=True)
  os.makedirs(spatiotemporal_maps_folder, exist_ok=True)

  def process_view(scenario_name: str, view: str):
    """Process a single view of a scenario.

    Args:
      scenario_name: Name of the scenario.
      view: View identifier.

    Returns:
      A dictionary of calculated metrics.
    """
    print('Processing scenario: ', scenario_name)
    real_path_v1 = os.path.join(real_folders, f'{scenario_name}_{view}_1.mp4')
    generated_path = os.path.join(
        generated_folders, f'{scenario_name}_{view}_1.mp4'
    )
    real_path_v2 = os.path.join(real_folders, f'{scenario_name}_{view}_2.mp4')
    binary_path_v1 = os.path.join(
        binary_real_folders, f'{scenario_name}_{view}_1.mp4'
    )
    binary_path_v2 = os.path.join(
        binary_real_folders, f'{scenario_name}_{view}_2.mp4'
    )
    binary_generated_path = os.path.join(
        binary_generated_folders, f'{scenario_name}_{view}_1.mp4'
    )
    # Load the first frame to determine the original target size
    real_v1_sample = load_and_resize_video(
        real_path_v1, int(fps_list * 3), int(fps_list * 3) + 1, normalize=False
    )
    if not real_v1_sample:
      return None  # Early exit if no frames are loaded
    target_size = (
        real_v1_sample[0].shape[1] // 4,
        real_v1_sample[0].shape[0] // 4,
    )

    real_v1_frames = load_and_resize_video(
        real_path_v1,
        int(fps_list * 3),
        int(fps_list * 3) + consider_frames,
        target_size,
    )
    generated_frames = load_and_resize_video(
        generated_path, start_frame, end_frame, target_size
    )
    real_v2_frames = load_and_resize_video(
        real_path_v2,
        int(fps_list * 3),
        int(fps_list * 3) + consider_frames,
        target_size,
    )
    binary_v1_frames = load_and_resize_video(
        binary_path_v1,
        int(fps_list * 3),
        int(fps_list * 3) + consider_frames,
        target_size,
        normalize=False,
    )
    binary_v2_frames = load_and_resize_video(
        binary_path_v2,
        int(fps_list * 3),
        int(fps_list * 3) + consider_frames,
        target_size,
        normalize=False,
    )
    binary_generated_frames = load_and_resize_video(
        binary_generated_path,
        start_frame,
        end_frame,
        target_size,
        normalize=False,
    )

    # Ensure binary masks are binary (0 or 1 pixel values)
    binary_v1_frames = [
        (mask > 127).astype(np.uint8) * 255 for mask in binary_v1_frames
    ]
    binary_v2_frames = [
        (mask > 127).astype(np.uint8) * 255 for mask in binary_v2_frames
    ]
    binary_generated_frames = [
        (mask > 127).astype(np.uint8) * 255 for mask in binary_generated_frames
    ]

    # Check if frames are loaded properly
    if (
        not generated_frames
        or not real_v2_frames
        or not real_v1_frames
        or not binary_v1_frames
        or not binary_v2_frames
    ):
      print(f'Skipping scenario {scenario_name}_{view} due to missing frames')
      return None

    # Create subdirectories for each model in spatial and spatiotemporal maps
    # folders
    model_name = generated_folders.split('/')[-1]
    model_spatial_maps_folder = os.path.join(spatial_maps_folder, model_name)
    model_spatiotemporal_maps_folder = os.path.join(
        spatiotemporal_maps_folder, model_name
    )
    os.makedirs(model_spatial_maps_folder, exist_ok=True)
    os.makedirs(model_spatiotemporal_maps_folder, exist_ok=True)

    # Calculate IOU for v1 and v2 masks
    iou_v1 = iou_per_frame(binary_v1_frames, binary_generated_frames)
    iou_v2 = iou_per_frame(binary_v2_frames, binary_generated_frames)

    # Collapse binary masks over time to calculate spatial IOU
    spatial_v1 = spatial_binary_masks(binary_v1_frames)
    spatial_v2 = spatial_binary_masks(binary_v2_frames)

    # Calculate spatial IOU for v1 and v2
    spatial_generated = spatial_binary_masks(binary_generated_frames)

    iou_v1_spatial = iou_per_frame([spatial_v1], [spatial_generated])[0]
    iou_v2_spatial = iou_per_frame([spatial_v2], [spatial_generated])[0]

    # Calculate spatial IOU
    spatial_iou = iou_per_frame([spatial_v1], [spatial_v2])[
        0
    ]  # Calculate IOU between collapsed spatial masks

    # Save spatial maps
    scenario_spatial_folder = os.path.join(
        model_spatial_maps_folder, scenario_name
    )
    os.makedirs(scenario_spatial_folder, exist_ok=True)
    cv2.imwrite(
        os.path.join(
            scenario_spatial_folder, f'{scenario_name}_{view}_v1_spatial.png'
        ),
        spatial_v1,
    )
    cv2.imwrite(
        os.path.join(
            scenario_spatial_folder, f'{scenario_name}_{view}_v2_spatial.png'
        ),
        spatial_v2,
    )

    # Calculate spatiotemporal presence IOU for v1 and v2 masks
    spatiotemporal_v1 = spatiotemporal_binary_mask(binary_v1_frames)
    spatiotemporal_v2 = spatiotemporal_binary_mask(binary_v2_frames)

    # Calculate spatiotemporal IOU
    max_spatiotemporal_v1 = np.sum(
        np.maximum(spatiotemporal_v1, binary_generated_frames)
    )
    if max_spatiotemporal_v1 != 0:
      iou_v1_spatiotemporal = np.sum(
          np.minimum(spatiotemporal_v1, binary_generated_frames)
      ) / max_spatiotemporal_v1
    else:
      iou_v1_spatiotemporal = 0

    max_spatiotemporal_v2 = np.sum(
        np.maximum(spatiotemporal_v2, binary_generated_frames)
    )
    if max_spatiotemporal_v2 != 0:
      iou_v2_spatiotemporal = np.sum(
          np.minimum(spatiotemporal_v2, binary_generated_frames)
      ) / max_spatiotemporal_v2
    else:
      iou_v2_spatiotemporal = 0

    # Check for empty spatiotemporal masks
    if spatiotemporal_v1.size == 0 or spatiotemporal_v2.size == 0:
      print(f'Empty spatiotemporal masks for scenario {scenario_name}_{view}')
      return None

    # Calculate variance for spatial and spatiotemporal IOUs
    variance_spatial = iou_per_frame(
        [spatial_v1], [spatial_v2]
    )  # Calculate per-frame spatial IOU between v1 and v2 masks
    spatiotemporal_v1_max = np.sum(
        np.maximum(spatiotemporal_v1, spatiotemporal_v2)
    )
    if spatiotemporal_v1_max != 0:
      variance_spatiotemporal = np.sum(
          np.minimum(spatiotemporal_v1, spatiotemporal_v2)
      ) / spatiotemporal_v1_max
    else:
      variance_spatiotemporal = 0

    # Calculate variance IOU
    variance_iou = iou_per_frame(
        binary_v1_frames, binary_v2_frames
    )  # Calculate per-frame IOU between v1 and v2 masks

    # Calculate variance MSE
    variance_mse = mse_per_frame(
        real_v1_frames, real_v2_frames
    )  # Calculate MSE between v1 and v2 frames

    # Print average variances for this scenario
    print(f"Average Variances for Scenario '{scenario_name}_{view}':")
    print(f' - Variance Spatial: {np.mean(variance_spatial):.4f}')
    print(f' - Variance Spatiotemporal: {variance_spatiotemporal:.4f}')
    print(f' - Variance IOU: {np.mean(variance_iou):.4f}')
    print(f' - Variance MSE: {np.mean(variance_mse):.4f}')

    # Return the result for this view
    return {
        f'v1_mse_{view}': mse_per_frame(real_v1_frames, generated_frames),
        f'v2_mse_{view}': mse_per_frame(real_v2_frames, generated_frames),
        f'spatial_iou_{view}': spatial_iou,  # Spatial IOU
        f'variance_spatial_{view}': np.mean(
            variance_spatial
        ),  # Variance for spatial IOU
        f'variance_spatiotemporal_{view}': (
            variance_spatiotemporal
        ),  # Variance for spatiotemporal IOU
        f'variance_iou_{view}': variance_iou,  # Variance IOU metric
        f'variance_mse_{view}': variance_mse,  # Variance MSE metric
        f'upper_bound_{view}': mse_per_frame(
            [real_v1_frames[int(3 * fps_list) - 1]] * int(consider_frames),
            real_v2_frames[-int(consider_frames) :],
        ),
        f'iou_v1_{view}': iou_v1,  # IOU for v1
        f'iou_v2_{view}': iou_v2,  # IOU for v2
        'spatial_iou_v1': iou_v1_spatial,  # Spatial IOU for v1
        'spatial_iou_v2': iou_v2_spatial,  # Spatial IOU for v2
        'spatiotemporal_iou_v1': (
            iou_v1_spatiotemporal
        ),  # Spatiotemporal IOU for v1
        'spatiotemporal_iou_v2': (
            iou_v2_spatiotemporal
        ),  # Spatiotemporal IOU for v2
    }

  # Iterate over all scenarios
  for real_file in os.listdir(real_folders):
    if real_file.endswith('.mp4'):
      parts = real_file.rsplit('_', 2)
      if len(parts) < 3:
        print(f'Unexpected filename format: {real_file}')
        continue
      scenario_name = '_'.join(parts[:-2])
      _, version = parts[-2:]
      version = version.split('.')[0]

      if scenario_name in processed_scenarios:
        continue

      if version == '1':
        processed_scenarios.add(scenario_name)
        progress_bar.update(1)
        scenario_result = {
            'scenario': scenario_name,
        }

        with pool.ThreadPool() as executor:
          futures = [
              executor.apply_async(process_view, (scenario_name, view))
              for view in ['A', 'B', 'C']
          ]
          for future in futures:
            view_result = future.get()
            if view_result:
              for key, value in view_result.items():
                if isinstance(value, list):
                  scenario_result[key] = [float(v) for v in value]
                else:
                  scenario_result[key] = value
        scenario_data.append(scenario_result)

  # Convert list to DataFrame and write to CSV
  if scenario_data:
    df = pd.DataFrame(scenario_data)
    df.to_csv(csv_file_path, index=False)
  else:
    print('No data to write to CSV')


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Process videos and plot MSE.')
  parser.add_argument(
      '--real_folder',
      type=str,
      nargs='+',
      help='Paths to the folders containing real videos.',
  )
  parser.add_argument(
      '--generated_folder',
      type=str,
      nargs='+',
      help='Paths to the folders containing generated videos.',
  )
  parser.add_argument(
      '--binary_real_folder',
      type=str,
      nargs='+',
      help='Paths to the folders containing binary real videos.',
  )
  parser.add_argument(
      '--binary_generated_folders',
      type=str,
      nargs='+',
      help='Paths to the folders containing binary generated videos.',
  )
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
  parser.add_argument(
      '--video_time',
      type=str,
      nargs='+',
      choices=['first', 'last'],
      default=['last'],
      help=(
          'Choose whether to pick the first or last 5 seconds '
          'of the video. Must match the number of CSV files.'
      ),
  )

  args = parser.parse_args()
  print(args.real_folder, args.generated_folder)
  if len(args.real_folder) != len(args.generated_folder):
    raise ValueError(
        'The number of real folders must match the number of generated folders.'
    )
  if len(args.real_folder) != len(args.csv_files):
    raise ValueError(
        'The number of real folders must match the number of CSV files.'
    )
  if len(args.real_folder) != len(args.fps):
    raise ValueError(
        'The number of real folders must match the number of FPS values.'
    )
  if len(args.video_time) != len(args.csv_files):
    raise ValueError(
        'The number of video time options must match the number of CSV files.'
    )

  # Process the videos and update the CSV files
  for (
      real_folder,
      generated_folder,
      binary_real_folder,
      binary_generated_folder,
      csv_file,
      fps,
      video_time,
  ) in zip(
      args.real_folder,
      args.generated_folder,
      args.binary_real_folder,
      args.binary_generated_folders,
      args.csv_files,
      args.fps,
      args.video_time,
  ):
    print(
        f"Processing {csv_file} with FPS {fps}, video time '{video_time}' using"
        f" real folder '{real_folder}' and generated folder"
        f" '{generated_folder}'."
    )
    process_videos(
        real_folder,
        generated_folder,
        binary_real_folder,
        binary_generated_folder,
        csv_file,
        fps,
        video_time,
    )

  # Plot MSE values
  plot_metrics(args.csv_files, './physics-IQ-dataset/plots', args.fps)

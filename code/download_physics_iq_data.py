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


import os
import subprocess
import multiprocessing

# Ensure 'spawn' is the default start method for multiprocessing
multiprocessing.set_start_method("spawn", force=True)

def download_physics_iq_data(fps: str, dry_run: bool = False):
    """
    Downloads the Physics-IQ dataset based on the specified FPS.

    Args:
      fps: Desired FPS (e.g., '8', '16', '24', '30').
      dry_run: If True, only prints the commands without executing them.
    """
    valid_fps = ['8', '16', '24', '30']
    download_fps = [fps] if fps in valid_fps else []
    download_fps.append('30')  # Always download 30FPS data

    base_url = "gs://physics-iq-benchmark"  # Base GCS URL
    local_base_dir = "./physics-IQ-benchmark"  # Local base directory

    directories = {
        "full-videos": ["take-1", "take-2"],
        "split-videos/conditioning": download_fps,
        "split-videos/testing": download_fps,
        "switch-frames": None,
        "video-masks/real": download_fps
    }

    for directory, subdirs in directories.items():
        if directory == "full-videos":
            for take in subdirs:
                for fps_option in download_fps:
                    remote_path = f"{base_url}/{directory}/{take}/{fps_option}FPS"
                    local_path = os.path.join(local_base_dir, directory, take, f"{fps_option}FPS")
                    os.makedirs(local_path, exist_ok=True)
                    print(f"Preparing to download: {remote_path} to {local_path}")
                    if not dry_run:
                        try:
                            # Limit parallel downloads to avoid freezing
                            subprocess.run(
                                ["gsutil", "-m", "-o", "GSUtil:parallel_process_count=5", "cp", "-r", remote_path + "/*", local_path], check=True
                            )
                            print(f"Downloaded: {remote_path}")
                        except subprocess.CalledProcessError as e:
                            print(f"Failed to download: {remote_path}. Error: {e}")
        elif subdirs:
            for fps_option in subdirs:
                remote_path = f"{base_url}/{directory}/{fps_option}FPS"
                local_path = os.path.join(local_base_dir, directory, f"{fps_option}FPS")
                os.makedirs(local_path, exist_ok=True)
                print(f"Preparing to download: {remote_path} to {local_path}")
                if not dry_run:
                    try:
                        subprocess.run(
                            ["gsutil", "-m", "-o", "GSUtil:parallel_process_count=5", "cp", "-r", remote_path + "/*", local_path], check=True
                        )
                        print(f"Downloaded: {remote_path}")
                    except subprocess.CalledProcessError as e:
                        print(f"Failed to download: {remote_path}. Error: {e}")
        else:
            remote_path = f"{base_url}/{directory}"
            local_path = os.path.join(local_base_dir, directory)
            os.makedirs(local_path, exist_ok=True)
            print(f"Preparing to download: {remote_path} to {local_path}")
            if not dry_run:
                try:
                    subprocess.run(
                        ["gsutil", "-m", "-o", "GSUtil:parallel_process_count=5", "cp", "-r", remote_path + "/*", local_path], check=True
                    )
                    print(f"Downloaded: {remote_path}")
                except subprocess.CalledProcessError as e:
                    print(f"Failed to download: {remote_path}. Error: {e}")

    print("Download process complete.")


# Example usage
user_fps = input("Enter the desired FPS (e.g., 8, 16, 24, 30, other): ").strip()
download_physics_iq_data(user_fps)

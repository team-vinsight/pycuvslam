# Orbbec Gemini 2 RGBD SLAM Example

This example demonstrates how to use the NVIDIA cuVSLAM library with an Orbbec Gemini 2 camera for RGBD SLAM.

## Files

- `rgbd_slam.py`: The main script for running SLAM. It captures RGB and Depth frames from the camera, passes them to cuVSLAM, and visualizes the result.
- `gemini2_calibrated_config.yaml`: Configuration file containing camera calibration parameters (intrinsics, distortion, etc.).
- `visualizer.py`: Helper script for visualizing the SLAM output using the Rerun SDK.

## Requirements

- Python 3.10+
- `cuvslam` library
- `opencv-python`
- `pyorbbecsdk`
- `rerun-sdk`

## Usage

To run the RGBD SLAM with a Gemini 2 camera, use the following command (adjusting resolution and options as needed):

```bash
python3 rgbd_slam.py --config ./gemini2_calibrated_config.yaml --resolution 640x400 --enable-distortion --enhance-depth --fast-depth
```

### Options

- `--config`: Path to the camera configuration YAML file.
- `--resolution`: Camera resolution (e.g., `640x400`).
- `--enable-distortion`: Enable distortion correction using parameters from the config file.
- `--enhance-depth`: Enable depth enhancement (filtering).
- `--fast-depth`: Use faster depth enhancement parameters.
- `--no-viz`: Disable the Rerun visualizer.
- `--disable-observations`: Disable exporting observations (faster).
- `--detect-stationary`: Enable stationary detection to reduce drift when not moving.

## Output

- The script visualizes the tracking in real-time using Rerun (browser opens automatically).
- Trajectory is saved to `trajectory_gemini2_rgbd.txt`.
- Pose data is saved to `pose_data_gemini2_rgbd.txt`.

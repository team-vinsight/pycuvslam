# Quickstart: Running TUM RGB-D on PyCuVSLAM

This guide documents the specific steps to run the TUM Freiburg3 dataset using the `pycuvslam-env` environment.

## 1. Environment Setup

Ensure you are using the correct Python virtual environment given the system configuration:

```bash
# Set path to include your virtual environment bin
export PATH=~/pycuvslam-env/bin:$PATH
```

## 2. Dataset Preparation

Reflecting the setup performed for the **freiburg3_long_office_household** dataset:

1.  **Create Directory**:
    ```bash
    mkdir -p examples/tum/dataset
    ```

2.  **Download Dataset**:
    ```bash
    wget https://cvg.cit.tum.de/rgbd/dataset/freiburg3/rgbd_dataset_freiburg3_long_office_household.tgz -O examples/tum/dataset/rgbd_dataset_freiburg3_long_office_household.tgz
    ```

3.  **Extract**:
    ```bash
    tar -xzf examples/tum/dataset/rgbd_dataset_freiburg3_long_office_household.tgz -C examples/tum/dataset
    rm examples/tum/dataset/rgbd_dataset_freiburg3_long_office_household.tgz
    ```

4.  **Install Calibration File**:
    The generic dataset doesn't come with the specific rig file needed for PyCuVSLAM, so we copy the example one:
    ```bash
    cp examples/tum/freiburg3_rig.yaml examples/tum/dataset/rgbd_dataset_freiburg3_long_office_household/freiburg3_rig.yaml
    ```

## 3. Running the Visual Odometry

Navigate to the example directory and run the script. Ensure dependencies (`rerun-sdk`, `numpy`, etc.) are installed in your environment.

```bash
cd examples/tum

# Install requirements if not already done
pip install -r ../requirements.txt

# Run the tracker (ensure environment PATH is set as above)
python3 track_tum.py
```

## 4. Troubleshooting

-   **ModuleNotFoundError: No module named 'rerun'**:
    -   Make sure you are running the python from `~/pycuvslam-env/bin/python3`.
    -   Or `export PATH=~/pycuvslam-env/bin:$PATH` before running.

-   **RuntimeError: Failed to find Rerun Viewer executable**:
    -   The `rerun` executable lives in `~/pycuvslam-env/bin/rerun`. Ensure this folder is in your `$PATH`.

#
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
"""
Orbbec Gemini 2 Depth Camera RGBD Visual SLAM
This script demonstrates how to use cuVSLAM with Orbbec Gemini 2 camera for RGBD SLAM
"""
# Import system libraries
import sys
import time
from typing import List, Optional, Tuple
import argparse

# Import computer vision and numerical computing libraries
import cv2  # OpenCV - image processing
import numpy as np  # NumPy - numerical computing
import yaml  # YAML - configuration file parsing

# Import Orbbec camera SDK
from pyorbbecsdk import *

# Import NVIDIA cuVSLAM library
import cuvslam as vslam

# Add realsense folder to system path for importing visualizers
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'examples', 'realsense')))
# Add cuvslam to system path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'bin', 'aarch64')))
from visualizer import RerunVisualizer

# ==================== Constant Definitions ====================
WARMUP_FRAMES = 30  # Warmup frames - SLAM system needs some frames to initialize (reduced for faster startup)
IMAGE_JITTER_THRESHOLD_MS = 200 * 1e6  # Image jitter threshold (nanoseconds) - increased threshold for better tolerance
NUM_VIZ_CAMERAS = 2  # Number of visualization cameras - for displaying color and depth images
DEPTH_SCALE_FACTOR = 1000.0  # Depth scale factor - convert millimeters to meters (Gemini 2 depth unit is millimeters)
FRAME_WAIT_TIMEOUT_MS = 100  # Frame wait timeout (milliseconds) - reduced timeout for better responsiveness


def simple_frame_to_bgr(frame) -> Optional[np.ndarray]:
    """
    Convert Orbbec frame to BGR format numpy array
    
    Args:
        frame: Orbbec camera frame object
        
    Returns:
        Optional[np.ndarray]: BGR format image array, returns None if conversion fails
    """
    # Get frame width and height
    width = frame.get_width()
    height = frame.get_height()
    
    # Create numpy array from frame data
    data = np.frombuffer(frame.get_data(), dtype=np.uint8)
    
    # Check if data size is correct (should be width * height * 3 bytes)
    if data.size != width * height * 3:
        return None
    
    # Reshape 1D array to 3D image array (height, width, 3)
    image = data.reshape((height, width, 3))
    
    # Convert RGB format to BGR format (OpenCV uses BGR)
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def create_depth_visualization(depth_data: np.ndarray) -> np.ndarray:
    """
    Create depth image visualization effect (similar to test_camera.py)
    
    Args:
        depth_data: Depth data array (unit: millimeters)
        
    Returns:
        np.ndarray: Colored depth visualization image
    """
    # Set depth range (millimeters) - filter out too close and too far points
    # Adjusted range for better depth point coverage
    min_depth = 100   # Minimum depth 100mm (reduced from 150mm)
    max_depth = 3000  # Maximum depth 3000mm (increased from 2000mm)
    
    # Limit depth values to specified range
    depth_clipped = np.clip(depth_data, min_depth, max_depth)
    
    # Invert depth values (close points appear bright, far points appear dark)
    depth_inverted = max_depth - depth_clipped
    
    # Normalize depth values to 0-255 range
    depth_normalized = cv2.normalize(depth_inverted, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Apply MAGMA color mapping (red-yellow-white gradient)
    depth_vis = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_MAGMA)
    
    return depth_vis


def get_gemini2_camera_intrinsics(color_profile) -> dict:
    """
    Extract camera intrinsics from Orbbec Gemini 2 color stream configuration
    
    Args:
        color_profile: Orbbec color stream configuration object
        
    Returns:
        dict: Dictionary containing camera intrinsics
            - fx, fy: Focal length (pixels)
            - cx, cy: Principal point coordinates (pixels)
            - width, height: Image resolution
    """
    # Get camera intrinsics
    intrinsics = color_profile.get_intrinsic()
    
    # Return intrinsics dictionary
    return {
        'fx': intrinsics.fx,      # X direction focal length
        'fy': intrinsics.fy,      # Y direction focal length
        'cx': intrinsics.cx,      # X direction principal point
        'cy': intrinsics.cy,      # Y direction principal point
        'width': intrinsics.width,   # Image width
        'height': intrinsics.height  # Image height
    }


def create_gemini2_rig(intrinsics: dict, distortion_coeffs: Optional[List[float]] = None) -> vslam.Rig:
    """
    Create cuVSLAM Rig object for Orbbec Gemini 2 camera
    
    Args:
        intrinsics: Camera intrinsics dictionary
        distortion_coeffs: Distortion coefficients list [k1, k2, p1, p2, k3]
        
    Returns:
        vslam.Rig: cuVSLAM Rig object containing camera configuration
    """
    # Create camera object
    cam = vslam.Camera()
    
    # Set camera intrinsics
    cam.focal = (intrinsics['fx'], intrinsics['fy'])      # Focal length
    cam.principal = (intrinsics['cx'], intrinsics['cy'])  # Principal point
    cam.size = (intrinsics['width'], intrinsics['height']) # Image size
    
    # Set distortion model
    if distortion_coeffs is not None and any(abs(coeff) > 1e-6 for coeff in distortion_coeffs):
        # Use RadialTangential distortion model (cuVSLAM supported distortion model)
        try:
            cam.distortion = vslam.Distortion(vslam.Distortion.Model.RadialTangential)
            cam.distortion.coeffs = distortion_coeffs
            print(f"Using distortion correction: {distortion_coeffs}")
        except AttributeError:
            # If RadialTangential is not available, try other models
            try:
                cam.distortion = vslam.Distortion(vslam.Distortion.Model.Radial)
                cam.distortion.coeffs = distortion_coeffs[:2]  # Only use first two radial distortion coefficients
                print(f"Using radial distortion correction: {distortion_coeffs[:2]}")
            except AttributeError:
                # If none are supported, use pinhole model
                cam.distortion = vslam.Distortion(vslam.Distortion.Model.Pinhole)
                print("Distortion model not supported, using pinhole model (no distortion)")
    else:
        # Using pinhole model (no distortion)
        cam.distortion = vslam.Distortion(vslam.Distortion.Model.Pinhole)
        print("Using pinhole model (no distortion)")
    
    # Camera pose in Rig coordinate system (camera located at Rig origin)
    cam.rig_from_camera = vslam.Pose(
        rotation=[0, 0, 0, 1],  # Unit quaternion (w, x, y, z)
        translation=[0, 0, 0]    # Zero translation
    )
    
    # Create Rig containing single camera
    rig = vslam.Rig()
    rig.cameras = [cam]
    
    return rig


def setup_gemini2_pipeline(target_width: Optional[int] = None, target_height: Optional[int] = None) -> Tuple[Pipeline, Config, dict]:
    """
    Setup Orbbec Gemini 2 camera pipeline and get camera intrinsics
    
    Args:
        target_width: Target image width（None means use default/highest resolution）
        target_height: Target image height（None means use default/highest resolution）
    
    Returns:
        Tuple[Pipeline, Config, dict]: 
            - Pipeline: Orbbec camera pipeline object
            - Config: Camera configuration object
            - dict: Camera intrinsics dictionary
    """
    # Create camera configuration and pipeline objects
    config = Config()
    pipeline = Pipeline()
    
    # Get color stream configuration - Use same method as test_camera.py
    color_profile_list = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
    color_profile = None
    
    # If resolution is specified, find matching configuration
    if target_width is not None and target_height is not None:
        print(f"Looking for resolution {target_width}x{target_height} RGB configuration...")
        for cp in color_profile_list:
            if cp.get_format() == OBFormat.RGB:
                if cp.get_width() == target_width and cp.get_height() == target_height:
                    color_profile = cp
                    print(f"✅ Found matching resolution configuration: {target_width}x{target_height}")
                    break
        
        if color_profile is None:
            print(f"⚠️  No exact matching resolution found {target_width}x{target_height}")
            print("Available RGB resolutions:")
            for cp in color_profile_list:
                if cp.get_format() == OBFormat.RGB:
                    print(f"  - {cp.get_width()}x{cp.get_height()}")
            print("Will use default resolution...")
    
    # If specified resolution not found, use default RGB configuration
    if color_profile is None:
        for cp in color_profile_list:
            if cp.get_format() == OBFormat.RGB:
                color_profile = cp
                break
    
    if color_profile is None:
        print("Error: No RGB format color stream configuration found")
        sys.exit(-1)
    
    # Get depth stream configuration aligned with color stream - Use hardware D2C alignment
    hw_d2c_profile_list = pipeline.get_d2c_depth_profile_list(color_profile, OBAlignMode.HW_MODE)
    if len(hw_d2c_profile_list) == 0:
        print("Error: No D2C aligned depth stream configuration found")
        sys.exit(-1)
    hw_d2c_profile = hw_d2c_profile_list[0]
    
    # Enable stream configuration
    config.enable_stream(hw_d2c_profile)  # Enable depth stream
    config.enable_stream(color_profile)   # Enable color stream
    config.set_align_mode(OBAlignMode.HW_MODE)  # Set hardware alignment mode
    pipeline.enable_frame_sync()  # Enable frame synchronization
    
    # Start pipeline
    pipeline.start(config)
    
    # Get initial frame to extract intrinsics - With retry mechanism
    print("Getting initial frame to extract intrinsics...")
    frames = None
    for attempt in range(10):  # Try up to 10 times
        frames = pipeline.wait_for_frames(100)
        if frames is not None:
            color_frame = frames.get_color_frame()
            if color_frame is not None:
                print(f"Attempt{attempt + 1}successfully obtained initial frame")
                break
        print(f"Attempt{attempt + 1}attempt: No valid frame obtained, retrying...")
    
    if frames is None:
        print("Error: Unable to get frames from camera after 10 attempts")
        sys.exit(-1)
    
    color_frame = frames.get_color_frame()
    if color_frame is None:
        print("Error: Unable to get color frame")
        sys.exit(-1)
    
    # Extract camera intrinsics
    intrinsics = get_gemini2_camera_intrinsics(color_profile)
    
    # Print camera intrinsics information
    print(f"Gemini 2 camera intrinsics:")
    print(f"  Resolution: {intrinsics['width']}x{intrinsics['height']}")
    print(f"  Focal length: ({intrinsics['fx']:.2f}, {intrinsics['fy']:.2f})")
    print(f"  Principal point: ({intrinsics['cx']:.2f}, {intrinsics['cy']:.2f})")
    
    return pipeline, config, intrinsics


def load_camera_config(config_path: str) -> dict:
    """
    Load camera configuration from YAML file
    
    Args:
        config_path: Configuration file path
        
    Returns:
        dict: Camera configuration dictionary
        
    Exceptions:
        FileNotFoundError: Thrown when configuration file does not exist
    """
    # Check if configuration file exists
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    # Read and parse YAML configuration file
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def apply_depth_to_color_transform(depth_data: np.ndarray, transform: dict) -> np.ndarray:
    """
    Apply depth camera to color camera transformation (if needed)
    
    Args:
        depth_data: Original depth data
        transform: Depth to color transformation parameters
        
    Returns:
        np.ndarray: Transformed depth data
    """
    # Currently Orbbec SDK has handled D2C alignment，So this is mainly for completeness
    # If additional transformation processing is needed in the future，Can be implemented here
    return depth_data


def validate_depth_color_alignment(color_image: np.ndarray, depth_data: np.ndarray) -> bool:
    """
    Validate depth and color image alignment quality
    
    Args:
        color_image: Color image
        depth_data: Depth data
        
    Returns:
        bool: Whether alignment quality is good
    """
    # Check if image size matches
    if color_image.shape[:2] != depth_data.shape:
        print(f"Warning: Depth and color image size mismatch - Color: {color_image.shape[:2]}, Depth: {depth_data.shape}")
        return False
    
    # Check validity of depth data
    valid_depth_ratio = np.sum(depth_data > 0) / depth_data.size
    if valid_depth_ratio < 0.15:  # If valid depth points less than 15% (reduced threshold)
        print(f"Warning: Valid depth point ratio too low: {valid_depth_ratio:.2%}")
        return False
    
    return True


def suggest_performance_optimizations(avg_fps: float, valid_depth_ratio: float, frame_interval_ms: float) -> List[str]:
    """
    Suggest performance optimizations based on current metrics
    
    Args:
        avg_fps: Average FPS
        valid_depth_ratio: Ratio of valid depth points
        frame_interval_ms: Average frame interval in milliseconds
        
    Returns:
        List[str]: List of optimization suggestions
    """
    suggestions = []
    
    if avg_fps < 15:
        suggestions.append("🚀 Try --resolution 640x480 for maximum FPS improvement")
        suggestions.append("⚡ Use --fast-depth for faster depth processing")
        suggestions.append("🔧 Use --viz-skip-frames 3 to reduce visualization overhead")
    
    if valid_depth_ratio < 0.3:
        suggestions.append("📏 Improve lighting conditions for better depth sensing")
        suggestions.append("🎯 Ensure objects are within 0.5-5 meters range")
        suggestions.append("💡 Avoid reflective surfaces that affect depth perception")
    
    if frame_interval_ms > 100:
        suggestions.append("⏱️  Use --use-hardware-timestamp for better timing accuracy")
        suggestions.append("🔄 Try --disable-observations to reduce processing load")
        suggestions.append("🖥️  Close other applications to free up system resources")
    
    return suggestions


def enhance_depth_quality(depth_data: np.ndarray, fast_mode: bool = True) -> np.ndarray:
    """
    Enhance depth data quality
    
    Args:
        depth_data: Original depth data
        fast_mode: Fast mode（Use smaller filter kernel to improve performance）
        
    Returns:
        np.ndarray: Enhanced depth data
    """
    if fast_mode:
        # Fast mode: Only use 3x3 median filter, significantly improve performance
        depth_enhanced = cv2.medianBlur(depth_data.astype(np.uint16), 3)
    else:
        # Complete mode: Use larger filter kernel, better quality but slower speed
        # Apply median filter to remove noise
        depth_enhanced = cv2.medianBlur(depth_data.astype(np.uint16), 5)
        
        # For depth data, use Gaussian filter instead of bilateral filter（Because bilateral filter does not support uint16）
        # First convert to float32 for filtering, then convert back to uint16
        depth_float = depth_enhanced.astype(np.float32)
        depth_filtered = cv2.GaussianBlur(depth_float, (5, 5), 1.0)
        depth_enhanced = depth_filtered.astype(np.uint16)
    
    return depth_enhanced


def main() -> None:
    """
    Functions:
    1. Parse command line arguments
    2. Setup camera pipeline
    3. Initialize SLAM tracker
    4. Run real-time tracking loop
    5. Save trajectory data
    """
    # Create command line argument parser
    parser = argparse.ArgumentParser(description='Orbbec Gemini 2 RGBD Visual SLAM')
    
    # Add command line arguments
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Camera configuration YAML file path（Use calibration parameters if provided）'
    )
    parser.add_argument(
        '--undistort',
        action='store_true',
        help='Use calibration parameters for distortion correction'
    )
    parser.add_argument(
        '--no-viz',
        action='store_true',
        help='Disable visualization（Useful when Rerun server is not available）'
    )
    parser.add_argument(
        '--enable-distortion',
        action='store_true',
        help='Enable distortion correction（Use distortion coefficients from calibration file）'
    )
    parser.add_argument(
        '--enhance-depth',
        action='store_true',
        help='Enable depth data quality enhancement（Filtering and noise removal）'
    )
    parser.add_argument(
        '--viz-skip-frames',
        type=int,
        default=1,
        help='Visualization skip frames（For example: 2 means visualize every other frame to improve performance）'
    )
    parser.add_argument(
        '--fast-depth',
        action='store_true',
        help='Use fast depth enhancement mode (3x3 filter instead of 5x5+Gaussian, improve performance)'
    )
    parser.add_argument(
        '--disable-observations',
        action='store_true',
        help='Disable observation export（Maximize performance, but visualization will not show feature points）'
    )
    parser.add_argument(
        '--resolution',
        type=str,
        default=None,
        help='Camera resolution (Format: WIDTHxHEIGHT, For example: 640x480, 1280x720)。Common: 640x480(Fastest), 1280x720(Balanced), 1920x1080(Default)'
    )
    parser.add_argument(
        '--list-resolutions',
        action='store_true',
        help='List all supported resolutions and exit'
    )
    parser.add_argument(
        '--use-hardware-timestamp',
        action='store_true',
        help='Use camera hardware timestamp instead of system timestamp（Improve time accuracy）'
    )
    parser.add_argument(
        '--diagnose-timestamps',
        action='store_true',
        help='Enable timestamp diagnosis mode（Display detailed frame interval statistics）'
    )
    parser.add_argument(
        '--camera-timeout',
        type=int,
        default=FRAME_WAIT_TIMEOUT_MS,
        help=f'Camera frame wait timeout（milliseconds），Default: {FRAME_WAIT_TIMEOUT_MS}ms'
    )
    parser.add_argument(
        '--detect-stationary',
        action='store_true',
        help='Enable stationary detection（Suppress pose updates when camera is stationary, reduce drift）'
    )
    
    # Parse command line arguments
    args = parser.parse_args()
    
    # If user wants to list all supported resolutions
    if args.list_resolutions:
        print("Querying supported resolutions...")
        try:
            from pyorbbecsdk import Pipeline, OBSensorType, OBFormat
            pipeline = Pipeline()
            color_profile_list = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
            
            print("\nSupported RGB resolutions:")
            print("-" * 40)
            resolutions = []
            for cp in color_profile_list:
                if cp.get_format() == OBFormat.RGB:
                    width = cp.get_width()
                    height = cp.get_height()
                    res_str = f"{width}x{height}"
                    if res_str not in resolutions:
                        resolutions.append(res_str)
                        # Add performance suggestions
                        if width <= 640:
                            perf = "🚀 Fastest"
                        elif width <= 1280:
                            perf = "⚡ Fast"
                        elif width <= 1920:
                            perf = "⚖️  Balanced"
                        else:
                            perf = "🐢 Slower"
                        print(f"  {res_str:15s} {perf}")
            
            print("-" * 40)
            print(f"\nUsage: --resolution WIDTHxHEIGHT")
            print(f"Example: python {sys.argv[0]} --resolution 640x480")
            
        except Exception as e:
            print(f"Error: Unable to query resolution - {e}")
        sys.exit(0)
    
    # Parse resolution parameters
    target_width = None
    target_height = None
    if args.resolution:
        try:
            width_str, height_str = args.resolution.split('x')
            target_width = int(width_str)
            target_height = int(height_str)
            print(f"Will use resolution: {target_width}x{target_height}")
        except ValueError:
            print(f"Error: Invalid resolution format '{args.resolution}'")
            print(f"Correct format: WIDTHxHEIGHT (For example: 640x480)")
            sys.exit(-1)
    
    # Print program title
    print("="*60)
    print("Orbbec Gemini 2 RGBD Visual SLAM")
    print("="*60)
    
    # TrySetup camera pipeline（With retry mechanism）
    pipeline = None
    config = None
    intrinsics = None
    
    # Try up to 3 times to setup camera pipeline
    for attempt in range(3):
        try:
            print(f"Trying to setup camera pipeline（Attempt{attempt + 1}/3attempt）...")
            pipeline, config, intrinsics = setup_gemini2_pipeline(target_width, target_height)
            print("✅ Camera pipeline setup successful！")
            break
        except Exception as e:
            print(f"❌ Attempt{attempt + 1}attemptfailed: {e}")
            if attempt < 2:
                print("2seconds later retry...")
                time.sleep(2)
            else:
                print("All attempts failed。Please check：")
                print("1. Camera is properly connected")
                print("2. No other applications are using the camera")
                print("3. USB permissions are correct")
                sys.exit(-1)
    
    # If configuration file is provided, load calibration configuration
    calibrated_intrinsics = None
    distortion_coeffs = None
    depth_to_color_transform = None
    
    if args.config:
        print(f"Loading calibration configuration from path: {args.config}")
        calib_config = load_camera_config(args.config)
        
        # Use calibration values to override intrinsics
        calibrated_intrinsics = {
            'fx': calib_config['camera_matrix']['fx'],      # Calibrated X direction focal length
            'fy': calib_config['camera_matrix']['fy'],      # Calibrated Y direction focal length
            'cx': calib_config['camera_matrix']['cx'],      # Calibrated X direction principal point
            'cy': calib_config['camera_matrix']['cy'],      # Calibrated Y direction principal point
            'width': calib_config['image']['width'],        # Image width during calibration
            'height': calib_config['image']['height']       # Image height during calibration
        }
        
        # Load distortion coefficients（Only when distortion correction is enabled）
        if args.enable_distortion and 'distortion_coefficients' in calib_config:
            distortion_coeffs = [
                calib_config['distortion_coefficients']['k1'],
                calib_config['distortion_coefficients']['k2'],
                calib_config['distortion_coefficients']['p1'],
                calib_config['distortion_coefficients']['p2'],
                calib_config['distortion_coefficients']['k3']
            ]
            print(f"Distortion correction enabled")
        else:
            distortion_coeffs = None
            if not args.enable_distortion:
                print(f"Distortion correction disabled（Use --enable-distortion Enable）")
        
        # Load depth camera to color camera transformation parameters
        if 'depth_to_color_transform' in calib_config:
            depth_to_color_transform = calib_config['depth_to_color_transform']
        
        print(f"Using calibrated intrinsics:")
        print(f"  Resolution: {calibrated_intrinsics['width']}x{calibrated_intrinsics['height']}")
        print(f"  Focal length: ({calibrated_intrinsics['fx']:.2f}, {calibrated_intrinsics['fy']:.2f})")
        print(f"  Principal point: ({calibrated_intrinsics['cx']:.2f}, {calibrated_intrinsics['cy']:.2f})")
        if distortion_coeffs:
            print(f"  Distortion coefficients: {distortion_coeffs}")
        if depth_to_color_transform:
            print(f"  Depth-color transformation: Loaded")
    
    # Create camera Rig（Use calibration values if calibrated intrinsics available）
    # But maintain actual camera resolution for image processing
    if calibrated_intrinsics:
        # Use calibrated intrinsics but maintain actual camera resolution
        final_intrinsics = calibrated_intrinsics.copy()
        final_intrinsics['width'] = intrinsics['width']   # Use actual camera width
        final_intrinsics['height'] = intrinsics['height'] # Use actual camera height
        
        # Scale focal length and principal point proportionally
        scale_x = intrinsics['width'] / calibrated_intrinsics['width']
        scale_y = intrinsics['height'] / calibrated_intrinsics['height']
        
        final_intrinsics['fx'] *= scale_x  # Scale X direction focal length
        final_intrinsics['fy'] *= scale_y  # Scale Y direction focal length
        final_intrinsics['cx'] *= scale_x  # Scale X direction principal point
        final_intrinsics['cy'] *= scale_y  # Scale Y direction principal point
        
        print(f"Intrinsics scaled for actual resolution:")
        print(f"  Resolution: {final_intrinsics['width']}x{final_intrinsics['height']}")
        print(f"  Focal length: ({final_intrinsics['fx']:.2f}, {final_intrinsics['fy']:.2f})")
        print(f"  Principal point: ({final_intrinsics['cx']:.2f}, {final_intrinsics['cy']:.2f})")
        
        # Create Rig, use calibrated intrinsics and distortion coefficients
        rig = create_gemini2_rig(final_intrinsics, distortion_coeffs)
    else:
        # If no calibration file, use camera default intrinsics
        rig = create_gemini2_rig(intrinsics)
    
    # Configure RGBD settings
    rgbd_settings = vslam.Tracker.OdometryRGBDSettings()
    rgbd_settings.depth_scale_factor = DEPTH_SCALE_FACTOR  # Convert millimeters to meters
    rgbd_settings.depth_camera_id = 0  # Depth camera ID（First camera）
    rgbd_settings.enable_depth_stereo_tracking = False  # Disable depth stereo tracking
    
    # If depth to color transformation parameters exist, can be applied here
    if depth_to_color_transform:
        print("Depth-color transformation parameters loaded, will be used to optimize RGBD alignment")
    
    # Configure tracker - use supported parameters（Optimize performance）
    cfg = vslam.Tracker.OdometryConfig(
        async_sba=True,  # Enable async bundle adjustment（Improve performance）
        enable_final_landmarks_export=False,  # Disable final landmark export（Improve performance）
        odometry_mode=vslam.Tracker.OdometryMode.RGBD,  # Set to RGBD odometry mode
        rgbd_settings=rgbd_settings,  # Apply RGBD settings
        use_gpu=True,  # Use GPU acceleration
        use_motion_model=True,  # Use motion model（Improve tracking stability）
        use_denoising=False,  # Disable denoising（We have our own depth enhancement, save computation）
        enable_observations_export=not args.disable_observations,  # Observation export（For visualizing feature points）
        enable_landmarks_export=False  # Disable landmark export（Improve performance）
    )
    
    # Initialize tracker and visualizer
    tracker = vslam.Tracker(rig, cfg)  # Create SLAM tracker
    
    # Create visualizer（Optional, only supports Rerun visualization）
    visualizer = None
    if not args.no_viz:
        # Try to use Rerun visualizer
        try:
            visualizer = RerunVisualizer(num_viz_cameras=NUM_VIZ_CAMERAS)
            print("✅ Rerun visualizer initialization successful")
        except Exception as e:
            print(f"⚠️  Rerun visualizer initialization failed: {e}")
            print("Continue running, but no visualization interface...")
            visualizer = None
    
    # Print tracker initialization information
    print(f"\ncuVSLAM tracker initialized, odometry mode: RGBD")
    print(f"Depth scale factor: {DEPTH_SCALE_FACTOR} (Millimeters to meters)")
    
    # Print performance optimization configuration
    print(f"\n⚡ Performance configuration:")
    print(f"  Resolution: {intrinsics['width']}x{intrinsics['height']}")
    print(f"  Camera timeout: {args.camera_timeout}ms")
    print(f"  Timestamp mode: {'Hardware timestamp' if args.use_hardware_timestamp else 'System timestamp'}")
    print(f"  Timestamp diagnosis: {'Enabled (detailed mode)' if args.diagnose_timestamps else 'Disabled'}")
    print(f"  Visualization skip frames: Every {args.viz_skip_frames} frames")
    if args.enhance_depth:
        depth_mode = "Fast mode (3x3)" if args.fast_depth else "Complete mode (5x5+Gaussian)"
        print(f"  Depth enhancement: Enabled [{depth_mode}]")
    else:
        print(f"  Depth enhancement: Disabled")
    print(f"  Distortion correction: {'Enabled' if (args.enable_distortion and distortion_coeffs) else 'Disabled'}")
    print(f"  Visualizer: Rerun")
    print(f"  Observation export: {'Disabled (maximum performance)' if args.disable_observations else 'Enabled'}")
    print(f"  Stationary detection: {'Enabled (suppress drift)' if args.detect_stationary else 'Disabled'}")
    
    if args.viz_skip_frames == 1 and not args.disable_observations and args.resolution is None:
        print(f"\n💡 Performance optimization tips:")
        print(f"  If frame rate is low, try the following options（From low to high impact）:")
        print(f"  --resolution 640x480     # Reduce resolution (maximum improvement!)")
        print(f"  --viz-skip-frames 3      # Visualize every 3 frames (slight improvement)")
        print(f"  --fast-depth             # Use fast depth enhancement (medium improvement)")
        print(f"  --disable-observations   # Disable feature point export（Significant improvement）")
        print(f"  --no-viz                 # Completely disable visualization (maximum improvement)")
        print(f"\n  💡 Use --list-resolutions to view all supported resolutions")
    
    # Tracking variable initialization
    frame_id = 0  # Frame counter
    prev_timestamp: Optional[int] = None  # Previous frame timestamp
    trajectory: List[np.ndarray] = []  # Trajectory point list（Position only）
    pose_data: List[dict] = []  # Complete pose data list（Position + rotation）
    start_time = time.time()  # Start time
    frame_drop_warnings = 0  # Frame loss warning counter
    
    # Timestamp diagnosis statistics
    timestamp_intervals = []  # Timestamp interval list（For statistics）
    hardware_timestamp_base = None  # Hardware timestamp baseline
    last_frame_time = time.time()  # Previous frame system time（For calculating actual FPS）
    
    # Stationary detection
    stationary_threshold = 0.001  # 1mm position change threshold
    stationary_count = 0  # Consecutive stationary frames
    last_position = None  # Previous frame position
    is_stationary = False  # Currently stationary
    
    # Performance monitoring
    performance_monitor = {
        'frame_times': [],
        'processing_times': [],
        'fps_history': [],
        'last_optimization': 0
    }
    
    # Print start information and usage tips
    print("\n" + "="*60)
    print("Starting RGBD SLAM...")
    print("="*60)
    print("\n💡 Better tracking effect tips:")
    print("  1. Ensure color camera has good lighting conditions")
    print("  2. Avoid reflective surfaces affecting depth perception")
    print("  3. Move camera slowly and smoothly")
    print("  4. Keep objects within 0.5-5 meters for best depth quality")
    print("\nPress Ctrl+C to stop and save trajectory\n")
    
    try:
        # Main tracking loop
        while True:
            # Record frame acquisition start time
            frame_acquire_start = time.time()
            
            # Wait for frame data（Use configured timeout time）
            frames = pipeline.wait_for_frames(args.camera_timeout)
            if frames is None:
                continue  # If no frames obtained, continue next loop
            
            # Get color frames and depth frames
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            
            # If any frame is empty, skip this frame（Silently skip, similar to test_camera.py）
            if color_frame is None or depth_frame is None:
                continue
            
            # Convert frames to numpy array（Same method as test_camera.py）
            color_image = simple_frame_to_bgr(color_frame)
            if color_image is None:
                continue  # If conversion fails, silently skip this frame
            
            # Get depth data（Same method as test_camera.py）
            depth_height = depth_frame.get_height()  # Depth image height
            depth_width = depth_frame.get_width()    # Depth image width
            depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16).reshape((depth_height, depth_width))
            
            # Validate depth-color alignment quality
            if not validate_depth_color_alignment(color_image, depth_data):
                continue  # If alignment quality is poor, skip this frame
            
            # Enhance depth data quality（If enabled）
            if args.enhance_depth:
                depth_data = enhance_depth_quality(depth_data, fast_mode=args.fast_depth)
            
            # Apply depth to color transformation（If configured）
            if depth_to_color_transform:
                depth_data = apply_depth_to_color_transform(depth_data, depth_to_color_transform)
            
            # Generate timestamp（nanoseconds）
            if args.use_hardware_timestamp:
                # Try to use camera hardware timestamp
                try:
                    # Get color frame hardware timestamp（microseconds）
                    hw_timestamp_us = color_frame.get_timestamp()
                    
                    # Initialize hardware timestamp baseline
                    if hardware_timestamp_base is None:
                        hardware_timestamp_base = hw_timestamp_us
                    
                    # Convert to relative timestamp（nanoseconds）
                    timestamp_ns = int((hw_timestamp_us - hardware_timestamp_base) * 1000)
                except Exception as e:
                    # If hardware timestamp unavailable, fallback to system time
                    if frame_id == 0:
                        print(f"⚠️  Hardware timestamp unavailable, using system time: {e}")
                    timestamp_ns = int((time.time() - start_time) * 1e9)
            else:
                # UseSystem timestamp
                timestamp_ns = int((time.time() - start_time) * 1e9)
            
            # Calculate actual frame interval（For FPS statistics）
            current_frame_time = time.time()
            actual_frame_interval_ms = (current_frame_time - last_frame_time) * 1000
            last_frame_time = current_frame_time
            
            # Check timestamp difference with previous frame
            if prev_timestamp is not None:
                timestamp_diff = timestamp_ns - prev_timestamp
                timestamp_intervals.append(timestamp_diff / 1e6)  # Save interval（milliseconds）
                
                # Diagnosis mode: display detailed information
                if args.diagnose_timestamps and frame_id > WARMUP_FRAMES:
                    print(f"[frames {frame_id}] Timestamp interval: {timestamp_diff/1e6:.2f}ms, "
                          f"Actual interval: {actual_frame_interval_ms:.2f}ms, "
                          f"Real-time FPS: {1000/actual_frame_interval_ms:.1f}")
                
                # Normal mode: only warn when exceeding threshold
                if timestamp_diff > IMAGE_JITTER_THRESHOLD_MS:
                    frame_drop_warnings += 1
                    # Only display every 10 warnings to reduce information redundancy
                    if frame_drop_warnings % 10 == 1:
                        print(
                            f"⚠️  Timestamp interval too large: {timestamp_diff/1e6:.2f} ms "
                            f"(Threshold: {IMAGE_JITTER_THRESHOLD_MS/1e6:.2f} ms) "
                            f"[Actual FPS: {1000/actual_frame_interval_ms:.1f}] "
                            f"(#{frame_drop_warnings} times)"
                        )
            
            frame_id += 1  # Increment frame counter
            
            # Warmup specified number of frames
            if frame_id > WARMUP_FRAMES:
                # Prepare images for tracking
                images = [color_image]  # Color image list
                depths = [depth_data]   # Depth image list
                
                # Track current frame
                odom_pose_estimate, _ = tracker.track(
                    timestamp_ns, images=images, depths=depths
                )
                
                # Check if tracking succeeded
                if odom_pose_estimate.world_from_rig is None:
                    print(f"Warning: Tracking frame {frame_id} failed")
                    continue
                
                # Get current pose and observation data
                odom_pose = odom_pose_estimate.world_from_rig.pose
                current_position = np.array(odom_pose.translation)
                
                # Stationary detection
                if args.detect_stationary and last_position is not None:
                    position_change = np.linalg.norm(current_position - last_position)
                    
                    if position_change < stationary_threshold:
                        stationary_count += 1
                        if stationary_count > 30:  # Consecutive 30 frames stationary
                            is_stationary = True
                    else:
                        stationary_count = 0
                        is_stationary = False
                    
                    # If stationary detected, use first position（Suppress drift）
                    if is_stationary and len(trajectory) > 0:
                        # Use recent stable position instead of drifted position
                        current_position = last_position
                
                last_position = current_position.copy()
                
                trajectory.append(current_position)  # Add position to trajectory
                
                # Store complete pose data
                # cuVSLAM's odom_pose.rotation is in [x, y, z, w] order.
                # Convert and store as [w, x, y, z] for consistent usage elsewhere.
                raw_quat = odom_pose.rotation  # [x, y, z, w]
                qx, qy, qz, qw = raw_quat
                quat_wxyz = [qw, qx, qy, qz]

                pose_data.append({
                    'frame_id': frame_id,                    # framesID
                    'timestamp': timestamp_ns,               # Timestamp
                    'position': current_position,            # Position [x, y, z]
                    'rotation_quat': quat_wxyz,              # Rotation quaternion [w, x, y, z]
                    'stationary': is_stationary if args.detect_stationary else False  # Stationary flag
                })
                
                # Extract position and rotation information
                position = odom_pose.translation  # Position vector [x, y, z]

                # cuVSLAM returns quaternion in [x, y, z, w] order. Map to w,x,y,z for calculations.
                rotation_quat = odom_pose.rotation  # Quaternion [x, y, z, w]
                qx, qy, qz, qw = rotation_quat

                # Convert quaternion to Euler angles（Roll, pitch, yaw）
                import math
                w, x, y, z = qw, qx, qy, qz  # Quaternion components in w,x,y,z order
                
                # Roll (rotation around X axis)
                sinr_cosp = 2 * (w * x + y * z)
                cosr_cosp = 1 - 2 * (x * x + y * y)
                roll = math.atan2(sinr_cosp, cosr_cosp)
                
                # Pitch (rotation around Y axis)
                sinp = 2 * (w * y - z * x)
                if abs(sinp) >= 1:
                    pitch = math.copysign(math.pi / 2, sinp)  # If out of range, use 90 degrees
                else:
                    pitch = math.asin(sinp)
                
                # Yaw (rotation around Z axis)
                siny_cosp = 2 * (w * z + x * y)
                cosy_cosp = 1 - 2 * (y * y + z * z)
                yaw = math.atan2(siny_cosp, cosy_cosp)
                
                # Convert to degrees
                roll_deg = math.degrees(roll)   # Roll angle (degrees)
                pitch_deg = math.degrees(pitch) # Pitch angle (degrees)
                yaw_deg = math.degrees(yaw)     # Yaw angle (degrees)
                
                # Get observation data for visualization (if observation export is enabled)
                observations = [] if args.disable_observations else tracker.get_last_observations(0)
                
                # Store current timestamp for next iteration
                prev_timestamp = timestamp_ns
                
                # Visualize results (if enabled, only supports Rerun visualizer)
                # Use frame skipping to reduce visualization overhead for better performance
                if visualizer is not None and frame_id % args.viz_skip_frames == 0:
                    try:
                        # Rerun visualizer call（Consistent with run_rgbd.py）
                        # For RGBD, we only have one camera, so copy images and observation data
                        # Create depth visualization for second view
                        depth_vis = create_depth_visualization(depth_data)
                        
                        visualizer.visualize_frame(
                            frame_id=frame_id,                    # framesID
                            images=[images[0], depth_vis],        # Color image and depth visualization
                            pose=odom_pose,                       # Current pose
                            observations_main_cam=[observations, observations],  # Main camera observation data
                            trajectory=trajectory,                # Trajectory
                            timestamp=timestamp_ns                # Timestamp
                        )
                    except Exception as e:
                        # If visualization fails, silently continue running
                        if frame_id % 100 == 0:  # Print warning every 100 frames
                            print(f"⚠️  Visualization error: {e}")
                
                # Display status every 60 frames（Reduce print frequency to improve performance）
                if frame_id % 60 == 0:
                    elapsed = time.time() - start_time  # Elapsed time
                    fps = frame_id / elapsed if elapsed > 0 else 0  # Calculate FPS
                    num_features = len(observations)  # Number of feature points
                    
                    # Calculate valid depth ratio
                    valid_depth_ratio = np.sum(depth_data > 0) / depth_data.size
                    
                    # Calculate average frame interval
                    avg_interval_ms = np.mean(timestamp_intervals[-60:]) if len(timestamp_intervals) > 0 else 0
                    
                    # Feature quality indicator
                    feature_status = "🔴 LOW" if num_features < 30 else "🟡 OK" if num_features < 80 else "🟢 GOOD"
                    
                    # Stationary status indicator
                    motion_status = "🛑 Stationary" if is_stationary else "🚀 Moving"
                    
                    # Print detailed status information
                    status_line = f"📊 frames {frame_id}: {num_features} feature points {feature_status}, {fps:.1f} FPS"
                    if args.detect_stationary:
                        status_line += f" | {motion_status}"
                    print(status_line)
                    print(f"   📍 Position (XYZ): [{position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f}] meters")
                    print(f"   🔄 Rotation (RPY): Roll={roll_deg:.1f}°, Pitch={pitch_deg:.1f}°, Yaw={yaw_deg:.1f}°")
                    print(f"   🧭 Quaternion: w={w:.3f}, x={x:.3f}, y={y:.3f}, z={z:.3f}")
                    print(f"   📏 Depth coverage: {valid_depth_ratio:.1%} | Avg interval: {avg_interval_ms:.1f}ms")
                    
                    # Show performance suggestions if needed
                    if fps < 15 or valid_depth_ratio < 0.3 or avg_interval_ms > 100:
                        suggestions = suggest_performance_optimizations(fps, valid_depth_ratio, avg_interval_ms)
                        if suggestions:
                            print(f"   💡 Performance tips:")
                            for suggestion in suggestions[:3]:  # Show top 3 suggestions
                                print(f"      {suggestion}")
                    print()
            else:
                # During warmup, only show progress
                if frame_id % 10 == 0:
                    print(f"⏳ Warming up... frames {frame_id}/{WARMUP_FRAMES}")
    
    except KeyboardInterrupt:
        print("\nUser interrupted program")
    
    finally:
        # Cleanup and summary
        print("\n" + "="*60)
        print("RGBD SLAM session summary")
        print("="*60)
        print(f"Total processed frames: {frame_id}")
        print(f"Successful tracking: {len(trajectory)} poses")
        print(f"Frame loss warnings: {frame_drop_warnings}")
        if frame_id > WARMUP_FRAMES:
            success_rate = (len(trajectory) / (frame_id - WARMUP_FRAMES)) * 100
            print(f"Success rate: {success_rate:.1f}%")
        
        # Timestamp statistics
        if len(timestamp_intervals) > 0:
            import statistics
            avg_interval = statistics.mean(timestamp_intervals)
            min_interval = min(timestamp_intervals)
            max_interval = max(timestamp_intervals)
            median_interval = statistics.median(timestamp_intervals)
            stdev_interval = statistics.stdev(timestamp_intervals) if len(timestamp_intervals) > 1 else 0
            
            print(f"\n📊 Timestamp interval statistics:")
            print(f"  Average interval: {avg_interval:.2f} ms ({1000/avg_interval:.1f} FPS)")
            print(f"  Median interval: {median_interval:.2f} ms ({1000/median_interval:.1f} FPS)")
            print(f"  Minimum interval: {min_interval:.2f} ms ({1000/min_interval:.1f} FPS)")
            print(f"  Maximum interval: {max_interval:.2f} ms ({1000/max_interval:.1f} FPS)")
            print(f"  Standard deviation: {stdev_interval:.2f} ms")
            print(f"  Interval jitter: {(stdev_interval/avg_interval*100):.1f}%")
            
            # Analyze problems
            if avg_interval > 100:
                print(f"\n⚠️  Timestamp analysis:")
                print(f"  Average frame interval ({avg_interval:.1f}ms) Large, possible reasons：")
                print(f"  1. Slow processing speed（Try reducing resolution --resolution 640x480）")
                print(f"  2. Low camera frame rate（Check camera configuration）")
                print(f"  3. High CPU/GPU load（Close other programs）")
            
            if stdev_interval / avg_interval > 0.3:
                print(f"\n⚠️  Large timestamp jitter ({stdev_interval/avg_interval*100:.1f}%)，Possible reasons：")
                print(f"  1. Unstable system load")
                print(f"  2. Insufficient USB bandwidth")
                print(f"  3. Large visualization overhead（Try --viz-skip-frames or --no-viz）")
            
            if args.use_hardware_timestamp:
                print(f"\n✅ Hardware timestamp used")
            else:
                print(f"\n💡 Tip: Use --use-hardware-timestamp may improve time accuracy")
        
        # Save trajectory and pose data
        if len(trajectory) > 0:
            # Save simple trajectory（Position only）
            trajectory_array = np.array(trajectory)
            np.savetxt('trajectory_gemini2_rgbd.txt', trajectory_array, 
                       fmt='%.6f', delimiter=',',
                       header='x,y,z (meters)')
            
            # Save complete pose data（Position + rotation）
            with open('pose_data_gemini2_rgbd.txt', 'w') as f:
                f.write('# Frame_ID, Timestamp(ns), X(m), Y(m), Z(m), Qw, Qx, Qy, Qz\n')
                for pose in pose_data:
                    pos = pose['position']
                    quat = pose['rotation_quat']
                    f.write(f"{pose['frame_id']}, {pose['timestamp']}, "
                           f"{pos[0]:.6f}, {pos[1]:.6f}, {pos[2]:.6f}, "
                           f"{quat[0]:.6f}, {quat[1]:.6f}, {quat[2]:.6f}, {quat[3]:.6f}\n")
            
            print(f"\n✅ Data saved:")
            print(f"   📍 Trajectory: trajectory_gemini2_rgbd.txt ({len(trajectory)} poses)")
            print(f"   🎯 Complete pose: pose_data_gemini2_rgbd.txt ({len(pose_data)} poses)")
            
            # Calculate trajectory statistics
            if len(trajectory) > 1:
                distances = np.diff(trajectory_array, axis=0)
                total_distance = np.sum(np.linalg.norm(distances, axis=1))
                print(f"   📏 Total distance traveled: {total_distance:.2f} meters")
        else:
            print("\n⚠️  No trajectory data to save")
        
        # Stop camera pipeline and visualizer
        try:
            pipeline.stop()
            print("\nCamera released, program exiting...")
        except Exception as e:
            print(f"\nWarning: Error stopping camera pipeline: {e}")
        finally:
            # Close visualizer
            if visualizer is not None and hasattr(visualizer, 'close'):
                try:
                    visualizer.close()
                except Exception as e:
                    print(f"Error closing visualizer: {e}")
            print("="*60)


# Program entry point
if __name__ == "__main__":
    main()
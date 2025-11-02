import tensorflow as tf
import numpy as np
import rasterio
import rasterio.windows
import rasterio.features
import os
from skimage.transform import resize
from skimage.util import view_as_windows
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Re-define the U-Net model architecture (must be identical to training script)
def unet_model(input_size=(256, 256, 2)):
    inputs = tf.keras.layers.Input(input_size)

    # Encoder
    conv1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)

    # Bottleneck
    conv4 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)

    # Decoder
    up5 = tf.keras.layers.concatenate([tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv4), conv3], axis=-1)
    conv5 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(up5)
    conv5 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv5)

    up6 = tf.keras.layers.concatenate([tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv5), conv2], axis=-1)
    conv6 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(up6)
    conv6 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv6)

    up7 = tf.keras.layers.concatenate([tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv6), conv1], axis=-1)
    conv7 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(up7)
    conv7 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(conv7)

    outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(conv7) # Output a single channel mask

    model = models.Model(inputs=inputs, outputs=outputs)

    return model

# Function to load and preprocess original SAR images for prediction
def load_and_preprocess_sar_images(pre_flood_path, post_flood_path, target_shape):
    print(f"--- Debug: Entering load_and_preprocess_sar_images (simplified) ---")
    print(f"Debug: Target shape for resizing is: {target_shape}")
    with rasterio.open(pre_flood_path) as src:
        pre_flood_sar = src.read(1).astype(np.float32) # Convert to float32
        print(f"Debug: Loaded pre-flood raw SAR, shape: {pre_flood_sar.shape}, dtype: {pre_flood_sar.dtype}")
    with rasterio.open(post_flood_path) as src:
        post_flood_sar = src.read(1).astype(np.float32) # Convert to float32
        print(f"Debug: Loaded post-flood raw SAR, shape: {post_flood_sar.shape}, dtype: {post_flood_sar.dtype}")
    
    # Debug: Ensure both raw SAR images have the same shape before further processing
    if pre_flood_sar.shape != post_flood_sar.shape:
        print("Debug: Raw SAR images have different shapes. Cropping to smallest common extent.")
        min_rows = min(pre_flood_sar.shape[0], post_flood_sar.shape[0])
        min_cols = min(pre_flood_sar.shape[1], post_flood_sar.shape[1])
        pre_flood_sar = pre_flood_sar[:min_rows, :min_cols]
        post_flood_sar = post_flood_sar[:min_rows, :min_cols]
        print(f"Debug: Cropped pre-flood SAR to: {pre_flood_sar.shape}")
        print(f"Debug: Cropped post-flood SAR to: {post_flood_sar.shape}")

    # At this point, pre_flood_sar and post_flood_sar have the same shape.
    # Now, resize them to target_shape if necessary.
    if pre_flood_sar.shape != target_shape:
        pre_flood_sar_resized = resize(pre_flood_sar, target_shape, anti_aliasing=True)
        print(f"Debug: Resized pre-flood SAR to target_shape: {pre_flood_sar_resized.shape}")
    else:
        pre_flood_sar_resized = pre_flood_sar
        print(f"Debug: Pre-flood SAR already matches target_shape: {pre_flood_sar_resized.shape}")

    if post_flood_sar.shape != target_shape:
        post_flood_sar_resized = resize(post_flood_sar, target_shape, anti_aliasing=True)
        print(f"Debug: Resized post-flood SAR to target_shape: {post_flood_sar_resized.shape}")
    else:
        post_flood_sar_resized = post_flood_sar
        print(f"Debug: Post-flood SAR already matches target_shape: {post_flood_sar_resized.shape}")

    print(f"--- Debug: Exiting load_and_preprocess_sar_images (returning resized images) ---")
    return pre_flood_sar_resized, post_flood_sar_resized

# Function to extract patches for prediction
def extract_prediction_patches(pre_image, post_image, patch_size, batch_size):
    h, w = pre_image.shape

    pre_patches_view = view_as_windows(pre_image, (patch_size, patch_size), step=patch_size)
    post_patches_view = view_as_windows(post_image, (patch_size, patch_size), step=patch_size)

    num_rows_patches = pre_patches_view.shape[0]
    num_cols_patches = pre_patches_view.shape[1]
    total_patches = num_rows_patches * num_cols_patches

    # Generate coords once
    all_coords = []
    for r_idx in range(num_rows_patches):
        for c_idx in range(num_cols_patches):
            i = r_idx * patch_size
            j = c_idx * patch_size
            all_coords.append((i, j))

    # Iterate and yield batches
    current_batch_patches = []
    current_batch_coords = []
    for idx in range(total_patches):
        r_idx = idx // num_cols_patches
        c_idx = idx % num_cols_patches

        pre_patch = pre_patches_view[r_idx, c_idx]
        post_patch = post_patches_view[r_idx, c_idx]

        # Stack individual patch
        stacked_patch = np.stack([pre_patch, post_patch], axis=-1)

        current_batch_patches.append(stacked_patch)
        current_batch_coords.append(all_coords[idx])

        if len(current_batch_patches) == batch_size or idx == total_patches - 1:
            yield np.array(current_batch_patches), current_batch_coords
            current_batch_patches = []
            current_batch_coords = []

# Function to reconstruct full image from patches
def reconstruct_image(patches, coords, original_shape, patch_size):
    h, w = original_shape
    reconstructed_image = np.zeros(original_shape, dtype=np.float32)
    for idx, (i, j) in enumerate(coords):
        reconstructed_image[i:i+patch_size, j:j+patch_size] = patches[idx].squeeze()
    return reconstructed_image

# Function to export GeoTIFF (reused from process_sar.py)
def export_geotiff(data, profile, output_filepath):
    if data.dtype == bool:
        data = data.astype(np.uint8)

    output_profile = profile.copy()
    output_profile.update({
        'dtype': data.dtype,  # Ensure dtype matches the data
        'height': data.shape[0],
        'width': data.shape[1],
        'count': 1,  # Single band output
        'compress': 'lzw' # LZW compression
    })

    with rasterio.open(output_filepath, 'w', **output_profile) as dst:
        dst.write(data, 1)
    print(f"GeoTIFF exported to: {output_filepath}")

# Utility function to find SAR TIFF files (copied from process_sar.py)
def find_sar_tiff(data_dir, date_str):
    """Finds the first VV SAR file (.tif or .tiff) for a given date."""
    safe_dirs = list(data_dir.glob(f"*{date_str}*.SAFE"))
    if not safe_dirs:
        raise FileNotFoundError(f"No SAFE folder found for {date_str} in {data_dir}")
    
    for safe_dir in safe_dirs:
        meas_dir = safe_dir / "measurement"
        if not meas_dir.exists():
            continue
        files = list(meas_dir.rglob("*vv*.tif*"))  # matches .tif or .tiff
        if files:
            return files[0]
    
    raise FileNotFoundError(f"No VV .tif/.tiff found for {date_str} in {data_dir}")

# Function to crop image to AOI
def crop_image_to_aoi(image_data, image_profile, min_lon, min_lat, max_lon, max_lat):
    transform = image_profile['transform']
    crs = image_profile['crs']

    # Convert geographical coordinates of the AOI bounding box corners to pixel coordinates
    # Top-left corner of AOI in pixel coordinates
    aoi_row_tl, aoi_col_tl = rasterio.transform.rowcol(transform, min_lon, max_lat)
    # Bottom-right corner of AOI in pixel coordinates
    aoi_row_br, aoi_col_br = rasterio.transform.rowcol(transform, max_lon, min_lat)

    # Determine the min/max row/col to define the window
    # rasterio.windows.Window expects col_off, row_off, width, height
    # col_off is the starting column, row_off is the starting row
    # width = end_col - start_col, height = end_row - start_row
    
    # Ensure col_start is the smaller of the two column indices, and col_end is the larger
    col_start = int(min(aoi_col_tl, aoi_col_br))
    col_end = int(max(aoi_col_tl, aoi_col_br))
    
    # Ensure row_start is the smaller of the two row indices, and row_end is the larger
    row_start = int(min(aoi_row_tl, aoi_row_br))
    row_end = int(max(aoi_row_tl, aoi_row_br))

    # Clamp coordinates to image bounds
    col_start = max(0, col_start)
    row_start = max(0, row_start)
    col_end = min(image_profile['width'], col_end)
    row_end = min(image_profile['height'], row_end)

    # Check if the window is valid (i.e., width and height are positive)
    width = col_end - col_start
    height = row_end - row_start

    if width <= 0 or height <= 0:
        # This means the AOI does not overlap with the image or is too small
        # We should raise an error or return empty data/profile
        raise ValueError(f"Invalid crop window: AOI ({min_lon},{min_lat},{max_lon},{max_lat}) does not overlap with the image extent or is too small after clamping. Calculated window: col_start={col_start}, row_start={row_start}, width={width}, height={height}. Image bounds: width={image_profile['width']}, height={image_profile['height']}")

    window = rasterio.windows.Window(col_start, row_start, width, height)

    # Crop the image data
    # Handle multi-channel images
    if image_data.ndim == 3: # (height, width, channels)
        cropped_data = image_data[window.row_off:window.row_off + window.height,
                                  window.col_off:window.col_off + window.width, :]
    else: # (height, width)
        cropped_data = image_data[window.row_off:window.row_off + window.height,
                                  window.col_off:window.col_off + window.width]

    # Update the profile for the cropped image
    cropped_profile = image_profile.copy()
    cropped_profile.transform = rasterio.windows.transform(window, transform)
    cropped_profile.width = window.width
    cropped_profile.height = window.height

    return cropped_data, cropped_profile

if __name__ == "__main__":
    from dotenv import load_dotenv
    from pathlib import Path

    # Load environment variables from the project root .env file
    # This allows the script to find the DATA_DIR and RESULTS_DIR when run from within the container
    load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / '.env')

    # Define paths using environment variables for Docker compatibility
    DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
    RESULTS_DIR = Path(os.getenv("RESULTS_DIR", "results"))
    MODELS_DIR = Path("models")

    print("--- Locating required files ---")
    # Find SAR TIFF files dynamically using the robust find_sar_tiff function
    pre_flood_tiff_filepath = find_sar_tiff(DATA_DIR, "20210511")
    post_flood_tiff_filepath = find_sar_tiff(DATA_DIR, "20210604")
    print(f"Found pre-flood TIFF: {pre_flood_tiff_filepath}")
    print(f"Found post-flood TIFF: {post_flood_tiff_filepath}")

    model_path = MODELS_DIR / "unet_flood_detection_model.keras"
    output_ml_flood_map_filepath = RESULTS_DIR / "flood_map_unet_prediction.tif"
    change_detection_map_path = RESULTS_DIR / "flood_map_change_detection.tif"

    # Load the trained model
    print(f"\n--- Loading Model ---")
    print(f"Loading trained U-Net model from {model_path}...")
    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully.")

    # Determine target shape from the output of process_sar.py
    print(f"\n--- Preprocessing ---")
    print(f"Using {change_detection_map_path} as reference for shape and profile...")
    if not change_detection_map_path.exists():
        raise FileNotFoundError(f"Reference map not found: {change_detection_map_path}. Please run process_sar.py first.")
    
    with rasterio.open(change_detection_map_path) as src:
        target_shape = src.shape
        original_profile = src.profile

    # Load and preprocess SAR images (now returns resized raw images)
    print("Loading and preprocessing SAR images for prediction...")
    pre_flood_resized, post_flood_resized = load_and_preprocess_sar_images(pre_flood_tiff_filepath, post_flood_tiff_filepath, target_shape)

    # Normalize SAR data (moved from function to main for debugging)
    print("Debug: Starting normalization...")
    
    # Pre-flood normalization
    pre_min = pre_flood_resized.min()
    pre_max = pre_flood_resized.max()
    print(f"Debug: Pre-flood min: {pre_min}, max: {pre_max}")
    pre_range = pre_max - pre_min
    if pre_range == 0: pre_range = 1.0 # Avoid division by zero for flat images
    pre_flood_sar_normalized = (pre_flood_resized - pre_min) / pre_range
    print(f"Debug: Normalized pre-flood SAR, shape: {pre_flood_sar_normalized.shape}")
    del pre_flood_resized # Free up memory

    # Post-flood normalization
    post_min = post_flood_resized.min()
    post_max = post_flood_resized.max()
    print(f"Debug: Post-flood min: {post_min}, max: {post_max}")
    post_range = post_max - post_min
    if post_range == 0: post_range = 1.0 # Avoid division by zero for flat images
    post_flood_sar_normalized = (post_flood_resized - post_min) / post_range
    print(f"Debug: Normalized post-flood SAR, shape: {post_flood_sar_normalized.shape}")
    del post_flood_resized # Free up memory

    # Extract patches for prediction and predict in batches
    patch_size = 256 # Must match training patch size
    batch_size = 32 # Define a suitable batch size for prediction
    print(f"Extracting prediction patches of size {patch_size}x{patch_size} and predicting in batches of {batch_size}...")

    all_predicted_patches = []
    all_patch_coords = []

    for batch_of_patches, batch_of_coords in extract_prediction_patches(pre_flood_sar_normalized, post_flood_sar_normalized, patch_size, batch_size):
        # Make predictions for the current batch
        predicted_batch = model.predict(batch_of_patches)
        all_predicted_patches.append(predicted_batch)
        all_patch_coords.extend(batch_of_coords)

    # Concatenate all predicted patches
    prediction_patches_combined = np.concatenate(all_predicted_patches, axis=0)

    # Reconstruct the full flood map from predicted patches
    print("\n--- Reconstructing full flood map ---")
    reconstructed_ml_flood_map = reconstruct_image(prediction_patches_combined, all_patch_coords, target_shape, patch_size)
    print(f"Debug: Shape of reconstructed map after reconstruct_image(): {reconstructed_ml_flood_map.shape}")
    ml_binary_flood_map = (reconstructed_ml_flood_map > 0.5).astype(np.uint8)
    print(f"Debug: Shape of final binary map before export: {ml_binary_flood_map.shape}")

    # Export the full ML-predicted flood map as GeoTIFF
    print(f"\n--- Exporting Results ---")
    export_geotiff(ml_binary_flood_map, original_profile, output_ml_flood_map_filepath)

    # (Visualization part from the original script is omitted for brevity in this automated run,
    # but can be added back if needed for generating PNGs)

    print("\nML flood prediction complete.")
    print(f"Output saved to: {output_ml_flood_map_filepath}")


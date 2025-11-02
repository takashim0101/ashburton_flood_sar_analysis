import rasterio
import numpy as np
import os
from pathlib import Path
from skimage.transform import resize

def create_patches(image, mask, patch_size, output_dir, prefix):
    """
    Extracts patches from an image and its corresponding mask.
    """
    h, w, c = image.shape # Unpack height, width, and channels
    patch_count = 0
    for i in range(0, h - patch_size + 1, patch_size):
        for j in range(0, w - patch_size + 1, patch_size):
            img_patch = image[i:i+patch_size, j:j+patch_size, :] # Include channel dimension
            mask_patch = mask[i:i+patch_size, j:j+patch_size]

            # Only save patches that contain some flood area to focus training
            if np.any(img_patch > 0) and np.any(mask_patch > 0):
                # Save image patch
                img_patch_path = os.path.join(output_dir, "images", f"{prefix}_img_{patch_count:04d}.npy")
                np.save(img_patch_path, img_patch)

                # Save mask patch
                mask_patch_path = os.path.join(output_dir, "masks", f"{prefix}_mask_{patch_count:04d}.npy")
                np.save(mask_patch_path, mask_patch)
                patch_count += 1
    return patch_count

if __name__ == "__main__":
    # Define base paths for the Docker environment
    DATA_DIR = "/app/data"
    RESULTS_DIR = "/app/results"
    ML_DATA_DIR = "ml_data"

    print("Searching for input files...")
    data_path = Path(DATA_DIR)
    results_path = Path(RESULTS_DIR)

    try:
        # Use glob to find the specific files needed, making it more robust
        pre_flood_files = list(data_path.glob('**/S1A_IW_GRDH_1SDV_20210511T*.SAFE/measurement/s1a-iw-grd-vv-*.tiff'))
        post_flood_files = list(data_path.glob('**/S1A_IW_GRDH_1SDV_20210604T*.SAFE/measurement/s1a-iw-grd-vv-*.tiff'))
        
        if not pre_flood_files:
            raise FileNotFoundError("Pre-flood TIFF file not found in /app/data.")
        if not post_flood_files:
            raise FileNotFoundError("Post-flood TIFF file not found in /app/data.")

        pre_flood_tiff_filepath = pre_flood_files[0]
        post_flood_tiff_filepath = post_flood_files[0]
        flood_map_filepath = results_path / "flood_map_change_detection.tif"

        if not flood_map_filepath.exists():
            raise FileNotFoundError(f"Flood map file not found at {flood_map_filepath}")

        print(f"Found Pre-flood SAR: {pre_flood_tiff_filepath}")
        print(f"Found Post-flood SAR: {post_flood_tiff_filepath}")
        print(f"Found Flood Map: {flood_map_filepath}")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure that the raw SAR data exists and that process_sar.py has been run successfully.")
        exit()

    # Output directory for ML data
    train_dir = os.path.join(ML_DATA_DIR, "train")
    os.makedirs(os.path.join(train_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(train_dir, "masks"), exist_ok=True)

    patch_size = 256

    print("Loading SAR images and flood map...")
    with rasterio.open(pre_flood_tiff_filepath) as src:
        pre_flood_sar = src.read(1)
    with rasterio.open(post_flood_tiff_filepath) as src:
        post_flood_sar = src.read(1)
    with rasterio.open(flood_map_filepath) as src:
        flood_mask = src.read(1)

    # Resize SAR images to match the flood_mask dimensions for patching
    if post_flood_sar.shape != flood_mask.shape:
        print(f"Resizing post_flood_sar from {post_flood_sar.shape} to {flood_mask.shape}")
        post_flood_sar_resized = resize(post_flood_sar, flood_mask.shape, anti_aliasing=True, preserve_range=True)
    else:
        post_flood_sar_resized = post_flood_sar

    if pre_flood_sar.shape != flood_mask.shape:
        print(f"Resizing pre_flood_sar from {pre_flood_sar.shape} to {flood_mask.shape}")
        pre_flood_sar_resized = resize(pre_flood_sar, flood_mask.shape, anti_aliasing=True, preserve_range=True)
    else:
        pre_flood_sar_resized = pre_flood_sar

    # Normalize SAR data (0-1 range)
    def normalize(array):
        min_val, max_val = array.min(), array.max()
        if max_val > min_val:
            return (array - min_val) / (max_val - min_val)
        return array

    pre_flood_sar_normalized = normalize(pre_flood_sar_resized)
    post_flood_sar_normalized = normalize(post_flood_sar_resized)

    # Combine into a multi-channel image
    combined_sar_image = np.stack([pre_flood_sar_normalized, post_flood_sar_normalized], axis=-1)

    print(f"Extracting patches of size {patch_size}x{patch_size}...")
    total_patches = create_patches(combined_sar_image, flood_mask, patch_size, train_dir, "sar")
    print(f"Successfully extracted {total_patches} patches and saved to {train_dir}")

    print("ML data preparation complete.")
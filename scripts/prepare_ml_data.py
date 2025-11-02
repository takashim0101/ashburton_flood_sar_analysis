import rasterio
import numpy as np
import os

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

            # Only save patches that are not entirely empty or entirely full (optional, but good for training)
            # Or, save all for now and filter later
            # For combined SAR image, check if any SAR data is present
            if np.any(img_patch > 0) and np.any(mask_patch > 0): # More robust check for non-empty patches
                # Save image patch
                img_patch_path = os.path.join(output_dir, "images", f"{prefix}_img_{patch_count:04d}.npy")
                np.save(img_patch_path, img_patch)

                # Save mask patch
                mask_patch_path = os.path.join(output_dir, "masks", f"{prefix}_mask_{patch_count:04d}.npy")
                np.save(mask_patch_path, mask_patch)
                patch_count += 1
    return patch_count

if __name__ == "__main__":
    # Define paths
    pre_flood_tiff_filepath = r"C:\Portfolio\ashburton_flood_sar_analysis\data\S1A_IW_GRDH_1SDV_20210511T173943_20210511T174008_037843_047769_94F3\S1A_IW_GRDH_1SDV_20210511T173943_20210511T174008_037843_047769_94F3.SAFE\measurement\s1a-iw-grd-vv-20210511t173943-20210511t174008-037843-047769-001.tiff"
    post_flood_tiff_filepath = r"C:\Portfolio\ashburton_flood_sar_analysis\data\S1A_IW_GRDH_1SDV_20210604T173944_20210604T174009_038193_0481EC_793C\S1A_IW_GRDH_1SDV_20210604T173944_20210604T174009_038193_0481EC_793C.SAFE\measurement\s1a-iw-grd-vv-20210604t173944-20210604t174009-038193-0481ec-001.tiff"
    flood_map_filepath = r"C:\Portfolio\ashburton_flood_sar_analysis\results\flood_map_change_detection.tif"

    # Output directory for ML data
    ml_data_dir = "ml_data"
    train_dir = os.path.join(ml_data_dir, "train")
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

    # Ensure all images are the same size (resize if necessary, though process_sar.py handles this)
    # For simplicity, we'll assume they are already aligned and processed to a common size
    # The flood_mask is already downsampled in process_sar.py, so we need to align SAR images to it
    # Or, we can downsample SAR images here to match flood_mask size for patching

    # Let's assume for now that post_flood_sar and flood_mask are aligned and processed
    # We need to ensure pre_flood_sar is also aligned to the same dimensions as post_flood_sar
    # For U-Net, input will be multi-channel (pre_flood, post_flood)
    # For simplicity, let's use post_flood_sar as the primary image for patching and flood_mask as its label

    # Downsample SAR images to match the flood_mask dimensions for patching
    # This is a simplified approach. A more robust solution would involve georeferenced patching.
    from skimage.transform import resize
    if post_flood_sar.shape != flood_mask.shape:
        print(f"Resizing post_flood_sar from {post_flood_sar.shape} to {flood_mask.shape} for patching.")
        post_flood_sar_resized = resize(post_flood_sar, flood_mask.shape, anti_aliasing=True)
    else:
        post_flood_sar_resized = post_flood_sar

    if pre_flood_sar.shape != flood_mask.shape:
        print(f"Resizing pre_flood_sar from {pre_flood_sar.shape} to {flood_mask.shape} for patching.")
        pre_flood_sar_resized = resize(pre_flood_sar, flood_mask.shape, anti_aliasing=True)
    else:
        pre_flood_sar_resized = pre_flood_sar

    # Normalize SAR data for better ML performance (e.g., 0-1 range)
    # Simple min-max normalization
    pre_flood_sar_normalized = (pre_flood_sar_resized - pre_flood_sar_resized.min()) / (pre_flood_sar_resized.max() - pre_flood_sar_resized.min())
    post_flood_sar_normalized = (post_flood_sar_resized - post_flood_sar_resized.min()) / (post_flood_sar_resized.max() - post_flood_sar_resized.min())

    # Combine pre-flood and post-flood SAR into a multi-channel image for U-Net input
    # Input for U-Net will be (height, width, channels)
    combined_sar_image = np.stack([pre_flood_sar_normalized, post_flood_sar_normalized], axis=-1)

    print(f"Extracting patches of size {patch_size}x{patch_size}...")
    total_patches = create_patches(combined_sar_image, flood_mask, patch_size, train_dir, "sar")
    print(f"Successfully extracted {total_patches} patches and saved to {train_dir}")

    print("ML data preparation complete.")

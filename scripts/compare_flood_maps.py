import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image # Using Pillow for image loading

# Increase the maximum image pixel limit for Pillow to handle large images
Image.MAX_IMAGE_PIXELS = 1000000000 # Set to 1 billion pixels or None to disable check

import rasterio # Import rasterio for reading GeoTIFFs

def create_comparison_figure(change_detection_geotiff, unet_prediction_geotiff, output_png):
    """
    Loads two flood map GeoTIFFs, normalizes them, and creates a colorized comparison figure.
    """
    try:
        with rasterio.open(change_detection_geotiff) as src_cd:
            data_cd = src_cd.read(1).astype(np.float32)
            if src_cd.nodata is not None:
                data_cd = np.ma.masked_equal(data_cd, src_cd.nodata)

        with rasterio.open(unet_prediction_geotiff) as src_unet:
            data_unet = src_unet.read(1).astype(np.float32)
            if src_unet.nodata is not None:
                data_unet = np.ma.masked_equal(data_unet, src_unet.nodata)

    except FileNotFoundError as e:
        print(f"Error loading GeoTIFF: {e}. Please ensure GeoTIFFs are generated.")
        return

    # Normalize data to 0-1 range
    def normalize_data(data_array):
        min_val = np.min(data_array)
        max_val = np.max(data_array)
        if max_val > min_val:
            return (data_array - min_val) / (max_val - min_val)
        return np.zeros_like(data_array, dtype=np.float32)

    normalized_cd = normalize_data(data_cd)
    normalized_unet = normalize_data(data_unet)

    # Ensure images are the same size (should be if from same source GeoTIFFs)
    if normalized_cd.shape != normalized_unet.shape:
        print("Warning: Input GeoTIFFs have different sizes. Resizing U-Net prediction to match Change Detection.")
        # This would require a more robust resizing for GeoTIFFs, but for display, simple resize might suffice
        # For now, assuming they are the same size as they come from the same source
        pass 

    fig, axes = plt.subplots(1, 2, figsize=(20, 10)) # Adjust figsize as needed
    cmap = plt.cm.Blues
    cmap.set_under('white', alpha=0) # Set values below vmin to transparent
    vmin_val = 0.1 # Minimum value for color mapping
    vmax_val = 1.0 # Maximum value for color mapping

    im0 = axes[0].imshow(normalized_cd, cmap=cmap, vmin=vmin_val, vmax=vmax_val)
    axes[0].set_title("Change Detection Flood Map")
    axes[0].axis('off')

    im1 = axes[1].imshow(normalized_unet, cmap=cmap, vmin=vmin_val, vmax=vmax_val)
    axes[1].set_title("U-Net Prediction Flood Map")
    axes[1].axis('off')

    plt.tight_layout()
    
    # Add a colorbar for the probability scale
    cbar = fig.colorbar(im0, ax=axes.ravel().tolist(), orientation='vertical', shrink=0.75)
    cbar.set_label('Flood Probability')
    cbar.set_ticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]) # Explicitly set ticks
    cbar.set_ticklabels(['0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0']) # Explicitly set labels

    plt.savefig(output_png, bbox_inches='tight', dpi=300) # Save with high DPI
    plt.close(fig)
    print(f"Successfully created comparison figure: {output_png}")

if __name__ == "__main__":
    change_detection_geotiff = "/app/results/flood_map_change_detection.tif"
    unet_prediction_geotiff = "/app/results/flood_map_unet_prediction.tif"
    output_comparison_png = "/app/results/Figure_Flood_Map_Comparison.png"

    if os.path.exists(change_detection_geotiff) and os.path.exists(unet_prediction_geotiff):
        create_comparison_figure(change_detection_geotiff, unet_prediction_geotiff, output_comparison_png)
    else:
        print("Error: One or both input GeoTIFFs not found. Please ensure they are generated first.")

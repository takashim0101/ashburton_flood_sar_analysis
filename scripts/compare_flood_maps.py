import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import rasterio

Image.MAX_IMAGE_PIXELS = None

def create_comparison_figure(change_detection_geotiff, unet_prediction_geotiff, base_sar_geotiff, output_png):
    """
    Loads two flood map GeoTIFFs and overlays them on a downsampled base SAR image,
    creating a colorized comparison figure with a red color scheme suitable for dark backgrounds.
    """
    try:
        with rasterio.open(change_detection_geotiff) as src_cd:
            data_cd = src_cd.read(1).astype(np.float32)
        with rasterio.open(unet_prediction_geotiff) as src_unet:
            data_unet = src_unet.read(1).astype(np.float32)
        with rasterio.open(base_sar_geotiff) as src_sar:
            data_sar = src_sar.read(1).astype(np.float32)
    except FileNotFoundError as e:
        print(f"Error loading GeoTIFF: {e}. Please ensure all GeoTIFFs are generated.")
        return

    # --- Downsample data for visualization ---
    downsample_factor = 10
    data_cd_ds = data_cd[::downsample_factor, ::downsample_factor]
    data_unet_ds = data_unet[::downsample_factor, ::downsample_factor]
    data_sar_ds = data_sar[::downsample_factor, ::downsample_factor]

    # --- Data Normalization ---
    def normalize_flood_data(data_array):
        min_val, max_val = np.min(data_array), np.max(data_array)
        return (data_array - min_val) / (max_val - min_val) if max_val > min_val else np.zeros_like(data_array)

    normalized_cd = normalize_flood_data(data_cd_ds)
    normalized_unet = normalize_flood_data(data_unet_ds)

    vmin_sar, vmax_sar = np.percentile(data_sar_ds[data_sar_ds > 0], [2, 98])

    # --- Plotting with Dark Background Style ---
    with plt.style.context('dark_background'):
        fig, axes = plt.subplots(1, 2, figsize=(20, 10))
        
        cmap = plt.get_cmap('Reds')
        cmap.set_under(alpha=0)

        vmin_flood = 0.1
        vmax_flood = 1.0

        # Plot 1: Change Detection Overlay
        axes[0].imshow(data_sar_ds, cmap='gray', vmin=vmin_sar, vmax=vmax_sar)
        im0 = axes[0].imshow(normalized_cd, cmap=cmap, vmin=vmin_flood, vmax=vmax_flood)
        axes[0].set_title("Change Detection Flood Map")
        axes[0].axis('off')

        # Plot 2: U-Net Prediction Overlay
        axes[1].imshow(data_sar_ds, cmap='gray', vmin=vmin_sar, vmax=vmax_sar)
        im1 = axes[1].imshow(normalized_unet, cmap=cmap, vmin=vmin_flood, vmax=vmax_flood)
        axes[1].set_title("U-Net Prediction Flood Map")
        axes[1].axis('off')

        plt.tight_layout()
        
        # Add a shared colorbar with percentage ticks and risk labels
        cbar = fig.colorbar(im0, ax=axes.ravel().tolist(), orientation='vertical', shrink=0.75)
        cbar.set_label('Flood Probability / Normalized Change (%)')
        
        ticks = np.linspace(vmin_flood, vmax_flood, 5)
        cbar.set_ticks(ticks)
        cbar.set_ticklabels([f'{int(t*100)}' for t in ticks])

        # Add High/Low risk labels
        cbar.ax.text(1.5, 0.95, 'High Risk', transform=cbar.ax.transAxes, va='center', ha='left', fontsize=10)
        cbar.ax.text(1.5, 0.05, 'Low Risk', transform=cbar.ax.transAxes, va='center', ha='left', fontsize=10)

        # Set figure facecolor to transparent for saving
        fig.patch.set_facecolor('none')
        plt.savefig(output_png, bbox_inches='tight', dpi=300, facecolor=fig.get_facecolor(), edgecolor='none')
        plt.close(fig)
        
    print(f"Successfully created comparison figure with overlay: {output_png}")

if __name__ == "__main__":
    change_detection_geotiff = "/app/results/flood_map_change_detection.tif"
    unet_prediction_geotiff = "/app/results/flood_map_unet_prediction.tif"
    base_sar_geotiff = "/app/results/post_flood_filtered.tif"
    output_comparison_png = "/app/results/Figure_Flood_Map_Comparison.png"

    required_files = [change_detection_geotiff, unet_prediction_geotiff, base_sar_geotiff]
    if all(os.path.exists(f) for f in required_files):
        create_comparison_figure(
            change_detection_geotiff, 
            unet_prediction_geotiff, 
            base_sar_geotiff, 
            output_comparison_png
        )
    else:
        print(f"Error: One or more input GeoTIFFs not found. Searched for: {required_files}")
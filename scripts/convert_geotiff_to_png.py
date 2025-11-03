import rasterio
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

def convert_geotiff_to_png(geotiff_path, base_sar_path, png_path, colorize=False):
    """
    Converts a single-band GeoTIFF to a PNG, optionally overlaying it on a base SAR image.
    """
    try:
        with rasterio.open(geotiff_path) as src:
            data_flood = src.read(1).astype(np.float32)
        
        if colorize:
            if not base_sar_path:
                print("Error: Base SAR path is required for colorized overlay.")
                return
            with rasterio.open(base_sar_path) as src_sar:
                data_sar = src_sar.read(1).astype(np.float32)
    except FileNotFoundError as e:
        print(f"Error loading GeoTIFF: {e}.")
        return

    if colorize:
        # --- Downsample for visualization ---
        downsample_factor = 10
        data_flood_ds = data_flood[::downsample_factor, ::downsample_factor]
        data_sar_ds = data_sar[::downsample_factor, ::downsample_factor]

        # --- Normalize ---
        min_val, max_val = np.min(data_flood_ds), np.max(data_flood_ds)
        normalized_data = (data_flood_ds - min_val) / (max_val - min_val) if max_val > min_val else np.zeros_like(data_flood_ds)
        
        vmin_sar, vmax_sar = np.percentile(data_sar_ds[data_sar_ds > 0], [2, 98])

        # --- Plotting with Dark Background Style ---
        with plt.style.context('dark_background'):
            fig, ax = plt.subplots(figsize=(11.5, 10)) # Adjusted figsize for colorbar
            
            # Draw base SAR image
            ax.imshow(data_sar_ds, cmap='gray', vmin=vmin_sar, vmax=vmax_sar)
            
            # Draw flood overlay
            cmap = plt.get_cmap('Reds')
            cmap.set_under(alpha=0)
            im = ax.imshow(normalized_data, cmap=cmap, vmin=0.1, vmax=1.0)
            
            # Add Title
            title = os.path.basename(geotiff_path).replace('.tif', '').replace('_', ' ').title()
            ax.set_title(title)
            ax.axis('off')

            # Add a colorbar with percentage ticks and risk labels
            cbar = fig.colorbar(im, ax=ax, shrink=0.8, orientation='vertical')
            cbar.set_label('Flood Probability (%)')
            
            ticks = np.linspace(0.1, 1.0, 5)
            cbar.set_ticks(ticks)
            cbar.set_ticklabels([f'{int(t*100)}' for t in ticks])

            # Add High/Low risk labels
            cbar.ax.text(2.0, 0.95, 'High Risk', transform=cbar.ax.transAxes, va='center', ha='left', fontsize=10)
            cbar.ax.text(2.0, 0.05, 'Low Risk', transform=cbar.ax.transAxes, va='center', ha='left', fontsize=10)

            plt.tight_layout(pad=0)
            
            # Set figure facecolor to transparent for saving
            fig.patch.set_facecolor('none')
            plt.savefig(png_path, bbox_inches='tight', pad_inches=0, dpi=300, facecolor=fig.get_facecolor(), edgecolor='none')
            plt.close(fig)

    else: # Grayscale conversion
        min_val, max_val = np.min(data_flood), np.max(data_flood)
        normalized_data = (data_flood - min_val) / (max_val - min_val) if max_val > min_val else np.zeros_like(data_flood)
        plt.imsave(png_path, (normalized_data * 255).astype(np.uint8), cmap='gray')
    
    print(f"Successfully converted {geotiff_path} to {png_path}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage (Grayscale): python convert_geotiff_to_png.py <input_geotiff> <output_png>")
        print("Usage (Colorized Overlay): python convert_geotiff_to_png.py <input_flood_geotiff> <base_sar_geotiff> <output_png> --colorize")
        sys.exit(1)

    colorize_flag = "--colorize" in sys.argv
    
    if colorize_flag:
        if len(sys.argv) != 5:
            print("Usage (Colorized Overlay): python convert_geotiff_to_png.py <input_flood_geotiff> <base_sar_geotiff> <output_png> --colorize")
            sys.exit(1)
        input_geotiff = sys.argv[1]
        base_sar_geotiff = sys.argv[2]
        output_png = sys.argv[3]
    else:
        input_geotiff = sys.argv[1]
        output_png = sys.argv[2]
        base_sar_geotiff = None

    if os.path.exists(input_geotiff) and (base_sar_geotiff is None or os.path.exists(base_sar_geotiff)):
        convert_geotiff_to_png(input_geotiff, base_sar_geotiff, output_png, colorize=colorize_flag)
    else:
        print(f"Error: Input file not found. Check paths for {input_geotiff} and/or {base_sar_geotiff}")

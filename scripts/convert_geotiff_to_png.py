import rasterio
import matplotlib.pyplot as plt
import numpy as np
import os
import sys # Import sys to access command-line arguments

def convert_geotiff_to_png(geotiff_path, png_path, colorize=False):
    """
    Converts a single-band GeoTIFF to a PNG image.
    If colorize is True, it uses a colormap (e.g., 'Blues') for visualization.
    """
    with rasterio.open(geotiff_path) as src:
        data = src.read(1) # Read the first band

        # Handle nodata values if present
        if src.nodata is not None:
            data = np.ma.masked_equal(data, src.nodata)

        # Normalize data to 0-1 range for colormap or 0-255 for grayscale
        data = data.astype(np.float32)
        min_val = np.min(data)
        max_val = np.max(data)
        
        if max_val > min_val:
            normalized_data = (data - min_val) / (max_val - min_val)
        else:
            normalized_data = np.zeros_like(data, dtype=np.float32) # Handle case where all values are the same

        if colorize:
            # Use a colormap for probability visualization
            # Set non-flood areas (or very low probability) to be transparent
            cmap = plt.cm.Blues
            cmap.set_under('white', alpha=0) # Set values below vmin to transparent
            
            # Use vmin to make sure low values are not colored
            plt.imsave(png_path, normalized_data, cmap=cmap, vmin=0.1, vmax=1.0)
        else:
            # For binary or grayscale, convert to 0-255 uint8
            normalized_data_uint8 = (normalized_data * 255).astype(np.uint8)
            plt.imsave(png_path, normalized_data_uint8, cmap='gray')
        
        print(f"Successfully converted {geotiff_path} to {png_path}")

if __name__ == "__main__":
    # Expects arguments: input_geotiff_path output_png_path [--colorize]
    if len(sys.argv) < 3:
        print("Usage: python convert_geotiff_to_png.py <input_geotiff_path> <output_png_path> [--colorize]")
        sys.exit(1)

    input_geotiff = sys.argv[1]
    output_png = sys.argv[2]
    colorize_flag = "--colorize" in sys.argv

    if os.path.exists(input_geotiff):
        convert_geotiff_to_png(input_geotiff, output_png, colorize=colorize_flag)
    else:
        print(f"Error: Input GeoTIFF not found at {input_geotiff}")

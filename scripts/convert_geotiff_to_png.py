import rasterio
import matplotlib.pyplot as plt
import numpy as np
import os

def convert_geotiff_to_png(geotiff_path, png_path):
    """
    Converts a single-band GeoTIFF to a PNG image.
    """
    with rasterio.open(geotiff_path) as src:
        data = src.read(1) # Read the first band

        # Normalize data to 0-255 for 8-bit PNG
        data = data.astype(np.float32)
        min_val = np.min(data)
        max_val = np.max(data)
        
        if max_val > min_val:
            normalized_data = ((data - min_val) / (max_val - min_val) * 255).astype(np.uint8)
        else:
            normalized_data = np.zeros_like(data, dtype=np.uint8) # Handle case where all values are the same

        # Save as PNG
        plt.imsave(png_path, normalized_data, cmap='gray')
        print(f"Successfully converted {geotiff_path} to {png_path}")

if __name__ == "__main__":
    input_geotiff = "/app/results/flood_map_unet_prediction.tif"
    output_png = "/app/results/flood_map_unet_prediction.png"
    
    if os.path.exists(input_geotiff):
        convert_geotiff_to_png(input_geotiff, output_png)
    else:
        print(f"Error: Input GeoTIFF not found at {input_geotiff}")

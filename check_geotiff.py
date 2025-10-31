import rasterio

output_filepath = "results/flood_map_change_detection.tif"

try:
    with rasterio.open(output_filepath) as src:
        print(f"CRS: {src.crs}")
        print(f"Transform: {src.transform}")
except Exception as e:
    print(f"Error reading GeoTIFF: {e}")

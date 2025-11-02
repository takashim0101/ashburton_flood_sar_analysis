# import rasterio
# import numpy as np
# from skimage.filters import median
# from skimage.morphology import disk
# from skimage.transform import resize
# import matplotlib.pyplot as plt
# import os
# import xml.etree.ElementTree as ET
# from rasterio.transform import from_gcps
# from rasterio.control import GroundControlPoint
# from skimage.filters import threshold_otsu
# 
# def read_sar_tiff(filepath):
#     """
#     Reads a SAR TIFF file and returns its data.
#     """
#     with rasterio.open(filepath) as src:
#         sar_data = src.read(1)  # Read the first band
#         profile = src.profile
#     return sar_data, profile
# 
# def parse_annotation_xml(xml_filepath):
#     """
#     Parses the Sentinel-1 annotation XML file to extract georeferencing information.
#     """
#     tree = ET.parse(xml_filepath)
#     root = tree.getroot()
# 
#     # Extract pixel spacing
#     image_info = root.find('.//imageInformation')
#     range_pixel_spacing = float(image_info.find('rangePixelSpacing').text)
#     azimuth_pixel_spacing = float(image_info.find('azimuthPixelSpacing').text)
# 
#     # Extract geolocation grid points
#     gcps = []
#     for grid_point in root.findall('.//geolocationGridPoint'):
#         line = int(grid_point.find('line').text)
#         pixel = int(grid_point.find('pixel').text)
#         latitude = float(grid_point.find('latitude').text)
#         longitude = float(grid_point.find('longitude').text)
#         gcps.append(GroundControlPoint(row=line, col=pixel, x=longitude, y=latitude))
# 
#     # Derive affine transform from GCPs
#     # Note: This is a simplified approach. For full accuracy, a more complex
#     # georeferencing model might be needed, potentially involving RPCs or more GCPs.
#     transform = from_gcps(gcps)
# 
#     # Determine CRS (assuming WGS84 from ellipsoidName)
#     crs = 'EPSG:4326' # WGS 84
# 
#     return transform, crs, range_pixel_spacing, azimuth_pixel_spacing
# 
# def radiometric_calibration(sar_data, calibration_factor=1.0):
#     """
#     Applies radiometric calibration to convert DNs to backscatter coefficients.
#     NOTE: For accurate calibration, actual calibration factors and incidence angle
#     from SAR product metadata (XML files) should be used.
#     This is a simplified placeholder.
#     """
#     # Example: Convert to power (linear scale) and then to dB
#     # For Sentinel-1 GRD, typically: sigma0_linear = (DN^2 + noise_power) / calibration_factor
#     # And then sigma0_db = 10 * log10(sigma0_linear)
#     # For simplicity, we'll just apply a linear scaling for now.
#     calibrated_data = sar_data * calibration_factor
#     return calibrated_data
# 
# def apply_speckle_filter(sar_data, radius=1):
#     """
#     Applies a median filter to reduce speckle noise.
#     """
#     footprint = disk(radius)
#     filtered_data = median(sar_data, footprint)
#     return filtered_data
# 
# def detect_flood(sar_data, threshold):
#     """
#     Detects flood areas using a simple thresholding method.
#     Pixels with values below the threshold are classified as water.
#     """
#     flood_map = sar_data < threshold
#     return flood_map
# 
# def perform_change_detection(ratio_image, threshold_ratio):
#     """
#     Performs change detection by applying a threshold to the ratio image.
#     Areas with a ratio below the threshold_ratio are classified as flooded.
#     """
#     change_map = ratio_image < threshold_ratio
#     return change_map
# 
# def visualize_results(original_data, filtered_data, flood_map, downsample_factor=10):
#     """
#     Visualizes the original SAR data, filtered data, and the detected flood map.
#     Downsamples images for display to reduce memory usage.
#     """
#     # Downsample data for visualization
#     original_data_downsampled = original_data[::downsample_factor, ::downsample_factor]
#     filtered_data_downsampled = filtered_data[::downsample_factor, ::downsample_factor]
#     flood_map_downsampled = flood_map[::downsample_factor, ::downsample_factor]
# 
#     fig, axes = plt.subplots(1, 4, figsize=(24, 6)) # Increased figure size and number of subplots
# 
#     # Adjust display range for original SAR data
#     vmin_original = np.percentile(original_data_downsampled, 2)
#     vmax_original = np.percentile(original_data_downsampled, 98)
# 
#     axes[0].imshow(original_data_downsampled, cmap='gray', vmin=vmin_original, vmax=vmax_original)
#     axes[0].set_title('Original SAR Data (Downsampled)')
#     axes[0].axis('off')
# 
#     # Adjust display range for filtered SAR data separately
#     vmin_filtered = np.percentile(filtered_data_downsampled, 2)
#     vmax_filtered = np.percentile(filtered_data_downsampled, 98)
# 
#     axes[1].imshow(filtered_data_downsampled, cmap='gray', vmin=vmin_filtered, vmax=vmax_filtered)
#     axes[1].set_title('Filtered SAR Data (Downsampled)')
#     axes[1].axis('off')
# 
#     # Create a custom colormap for the flood map: non-flood (0) as white, flood (1) as blue
#     from matplotlib.colors import ListedColormap
#     colors = ['white', 'blue'] # 0=white, 1=blue
#     cmap_flood = ListedColormap(colors)
# 
#     axes[2].imshow(flood_map_downsampled, cmap=cmap_flood, vmin=0, vmax=1)
#     axes[2].set_title('Detected Flood Map (Downsampled)')
#     axes[2].axis('off')
# 
#     # Overlay flood map on original SAR data
#     # Create a colormap for overlay: transparent for non-flood, blue for flood
#     colors_overlay = ['#FFFFFF00', 'blue'] # Transparent for 0, blue for 1
#     cmap_overlay = ListedColormap(colors_overlay)
# 
#     axes[3].imshow(original_data_downsampled, cmap='gray', vmin=vmin_original, vmax=vmax_original)
#     axes[3].imshow(flood_map_downsampled, cmap=cmap_overlay, vmin=0, vmax=1)
#     axes[3].set_title('Flood Overlay (Downsampled)')
#     axes[3].axis('off')
# 
#     plt.tight_layout()
#     plt.savefig(os.path.join("results", "Figure_1.png"))
#     # plt.show()
# 
# def analyze_ratio_histogram(ratio_image):
#     """
#     Analyzes the histogram of the ratio image to aid in threshold selection.
#     Clips the ratio image for better histogram visualization.
#     """
#     # Clip ratio_image for histogram plotting to a reasonable range
#     ratio_image_clipped = np.clip(ratio_image, 0, 5) # Clip to 0-5 for better visualization
# 
#     plt.figure(figsize=(8, 6))
#     plt.hist(ratio_image_clipped.flatten(), bins=100, color='gray', alpha=0.7)
#     plt.title('Histogram of Clipped Ratio Image (0-5)')
#     plt.xlabel('Ratio Value')
#     plt.ylabel('Frequency')
#     plt.grid(True)
#     plt.savefig(os.path.join("results", "histogram.png"))
#     plt.show()


import rasterio
import numpy as np
from skimage.filters import median
from skimage.morphology import disk
from skimage.transform import resize
import matplotlib.pyplot as plt
import os
import xml.etree.ElementTree as ET
from rasterio.transform import from_gcps
from rasterio.control import GroundControlPoint
from skimage.filters import threshold_otsu
import zipfile
import glob

# ----------------------------
# ZIP extraction function
# ----------------------------
def extract_sar_tiff_from_zip(zip_path, extract_to=None):
    """
    Automatically extracts TIFF files from a Sentinel-1 .SAFE directory inside a ZIP file.

    Args:
        zip_path (str): Path to the ZIP file.
        extract_to (str, optional): Folder to extract files to. Defaults to the ZIP's parent directory.

    Returns:
        str: Path to the first TIFF file found in the 'measurement' folder.
    """
    if extract_to is None:
        extract_to = os.path.dirname(zip_path)

    # Extract the ZIP archive
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

        # Determine the .SAFE folder name (top-level folder in ZIP)
        safe_dir_name = zip_ref.namelist()[0].split('/')[0]

    safe_dir_path = os.path.join(extract_to, safe_dir_name, "measurement")

    # Find all TIFF files inside the measurement folder
    tiff_files = glob.glob(os.path.join(safe_dir_path, "*.tiff"))
    if not tiff_files:
        raise FileNotFoundError(f"No TIFF files found in {safe_dir_path}")

    return tiff_files[0]  # Return the first TIFF found

# ----------------------------
# SAR functions (existing)
# ----------------------------
def read_sar_tiff(filepath):
    with rasterio.open(filepath) as src:
        sar_data = src.read(1)
        profile = src.profile
    return sar_data, profile

def parse_annotation_xml(xml_filepath):
    tree = ET.parse(xml_filepath)
    root = tree.getroot()
    image_info = root.find('.//imageInformation')
    range_pixel_spacing = float(image_info.find('rangePixelSpacing').text)
    azimuth_pixel_spacing = float(image_info.find('azimuthPixelSpacing').text)

    gcps = []
    for grid_point in root.findall('.//geolocationGridPoint'):
        line = int(grid_point.find('line').text)
        pixel = int(grid_point.find('pixel').text)
        latitude = float(grid_point.find('latitude').text)
        longitude = float(grid_point.find('longitude').text)
        gcps.append(GroundControlPoint(row=line, col=pixel, x=longitude, y=latitude))

    transform = from_gcps(gcps)
    crs = 'EPSG:4326'
    return transform, crs, range_pixel_spacing, azimuth_pixel_spacing

def radiometric_calibration(sar_data, calibration_factor=1.0):
    return sar_data * calibration_factor

def apply_speckle_filter(sar_data, radius=1):
    footprint = disk(radius)
    return median(sar_data, footprint)

def perform_change_detection(ratio_image, threshold_ratio):
    return ratio_image < threshold_ratio

def visualize_results(original_data, filtered_data, flood_map, downsample_factor=10):
    original_data_downsampled = original_data[::downsample_factor, ::downsample_factor]
    filtered_data_downsampled = filtered_data[::downsample_factor, ::downsample_factor]
    flood_map_downsampled = flood_map[::downsample_factor, ::downsample_factor]

    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    vmin_original = np.percentile(original_data_downsampled, 2)
    vmax_original = np.percentile(original_data_downsampled, 98)
    vmin_filtered = np.percentile(filtered_data_downsampled, 2)
    vmax_filtered = np.percentile(filtered_data_downsampled, 98)

    axes[0].imshow(original_data_downsampled, cmap='gray', vmin=vmin_original, vmax=vmax_original)
    axes[0].set_title('Original SAR Data (Downsampled)')
    axes[0].axis('off')

    axes[1].imshow(filtered_data_downsampled, cmap='gray', vmin=vmin_filtered, vmax=vmax_filtered)
    axes[1].set_title('Filtered SAR Data (Downsampled)')
    axes[1].axis('off')

    from matplotlib.colors import ListedColormap
    cmap_flood = ListedColormap(['white', 'blue'])
    axes[2].imshow(flood_map_downsampled, cmap=cmap_flood, vmin=0, vmax=1)
    axes[2].set_title('Detected Flood Map (Downsampled)')
    axes[2].axis('off')

    cmap_overlay = ListedColormap(['#FFFFFF00', 'blue'])
    axes[3].imshow(original_data_downsampled, cmap='gray', vmin=vmin_original, vmax=vmax_original)
    axes[3].imshow(flood_map_downsampled, cmap=cmap_overlay, vmin=0, vmax=1)
    axes[3].set_title('Flood Overlay (Downsampled)')
    axes[3].axis('off')

    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    plt.savefig(os.path.join("results", "Figure_1.png"))
    # plt.show()

# ----------------------------
# Main workflow
# ----------------------------
if __name__ == "__main__":
    pre_flood_zip = r"C:\Portfolio\ashburton_flood_sar_analysis\data\S1A_IW_GRDH_1SDV_20210511T173943_20210511T174008_037843_047769_94F3.zip"
    post_flood_zip = r"C:\Portfolio\ashburton_flood_sar_analysis\data\S1A_IW_GRDH_1SDV_20210604T173944_20210604T174009_038193_0481EC_793C.zip"

    pre_flood_tiff = extract_sar_tiff_from_zip(pre_flood_zip)
    post_flood_tiff = extract_sar_tiff_from_zip(post_flood_zip)

    pre_data, pre_profile = read_sar_tiff(pre_flood_tiff)
    post_data, post_profile = read_sar_tiff(post_flood_tiff)

    pre_cal = radiometric_calibration(pre_data, calibration_factor=0.1)
    post_cal = radiometric_calibration(post_data, calibration_factor=0.1)

    pre_filtered = apply_speckle_filter(pre_cal)
    post_filtered = apply_speckle_filter(post_cal)

    min_rows = min(pre_filtered.shape[0], post_filtered.shape[0])
    min_cols = min(pre_filtered.shape[1], post_filtered.shape[1])

    pre_filtered_cropped = pre_filtered[:min_rows, :min_cols]
    post_filtered_cropped = post_filtered[:min_rows, :min_cols]

    # Change detection
    ratio_image = post_filtered_cropped / np.where(pre_filtered_cropped == 0, 1e-6, pre_filtered_cropped)
    flood_map = perform_change_detection(ratio_image, threshold_ratio=0.8)

    # Visualization
    visualize_results(post_filtered_cropped, post_filtered_cropped, flood_map)


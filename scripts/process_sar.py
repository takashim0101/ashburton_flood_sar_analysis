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

def read_sar_tiff(filepath):
    """
    Reads a SAR TIFF file and returns its data.
    """
    with rasterio.open(filepath) as src:
        sar_data = src.read(1)  # Read the first band
        profile = src.profile
    return sar_data, profile

def parse_annotation_xml(xml_filepath):
    """
    Parses the Sentinel-1 annotation XML file to extract georeferencing information.
    """
    tree = ET.parse(xml_filepath)
    root = tree.getroot()

    # Extract pixel spacing
    image_info = root.find('.//imageInformation')
    range_pixel_spacing = float(image_info.find('rangePixelSpacing').text)
    azimuth_pixel_spacing = float(image_info.find('azimuthPixelSpacing').text)

    # Extract geolocation grid points
    gcps = []
    for grid_point in root.findall('.//geolocationGridPoint'):
        line = int(grid_point.find('line').text)
        pixel = int(grid_point.find('pixel').text)
        latitude = float(grid_point.find('latitude').text)
        longitude = float(grid_point.find('longitude').text)
        gcps.append(GroundControlPoint(row=line, col=pixel, x=longitude, y=latitude))

    # Derive affine transform from GCPs
    # Note: This is a simplified approach. For full accuracy, a more complex
    # georeferencing model might be needed, potentially involving RPCs or more GCPs.
    transform = from_gcps(gcps)

    # Determine CRS (assuming WGS84 from ellipsoidName)
    crs = 'EPSG:4326' # WGS 84

    return transform, crs, range_pixel_spacing, azimuth_pixel_spacing

def radiometric_calibration(sar_data, calibration_factor=1.0):
    """
    Applies radiometric calibration to convert DNs to backscatter coefficients.
    NOTE: For accurate calibration, actual calibration factors and incidence angle
    from SAR product metadata (XML files) should be used.
    This is a simplified placeholder.
    """
    # Example: Convert to power (linear scale) and then to dB
    # For Sentinel-1 GRD, typically: sigma0_linear = (DN^2 + noise_power) / calibration_factor
    # And then sigma0_db = 10 * log10(sigma0_linear)
    # For simplicity, we'll just apply a linear scaling for now.
    calibrated_data = sar_data * calibration_factor
    return calibrated_data

def apply_speckle_filter(sar_data, radius=1):
    """
    Applies a median filter to reduce speckle noise.
    """
    footprint = disk(radius)
    filtered_data = median(sar_data, footprint)
    return filtered_data

def detect_flood(sar_data, threshold):
    """
    Detects flood areas using a simple thresholding method.
    Pixels with values below the threshold are classified as water.
    """
    flood_map = sar_data < threshold
    return flood_map

def perform_change_detection(ratio_image, threshold_ratio):
    """
    Performs change detection by applying a threshold to the ratio image.
    Areas with a ratio below the threshold_ratio are classified as flooded.
    """
    change_map = ratio_image < threshold_ratio
    return change_map

def visualize_results(original_data, filtered_data, flood_map, downsample_factor=10):
    """
    Visualizes the original SAR data, filtered data, and the detected flood map.
    Downsamples images for display to reduce memory usage.
    """
    # Downsample data for visualization
    original_data_downsampled = original_data[::downsample_factor, ::downsample_factor]
    filtered_data_downsampled = filtered_data[::downsample_factor, ::downsample_factor]
    flood_map_downsampled = flood_map[::downsample_factor, ::downsample_factor]

    fig, axes = plt.subplots(1, 4, figsize=(24, 6)) # Increased figure size and number of subplots

    # Adjust display range for original SAR data
    vmin_original = np.percentile(original_data_downsampled, 2)
    vmax_original = np.percentile(original_data_downsampled, 98)

    axes[0].imshow(original_data_downsampled, cmap='gray', vmin=vmin_original, vmax=vmax_original)
    axes[0].set_title('Original SAR Data (Downsampled)')
    axes[0].axis('off')

    # Adjust display range for filtered SAR data separately
    vmin_filtered = np.percentile(filtered_data_downsampled, 2)
    vmax_filtered = np.percentile(filtered_data_downsampled, 98)

    axes[1].imshow(filtered_data_downsampled, cmap='gray', vmin=vmin_filtered, vmax=vmax_filtered)
    axes[1].set_title('Filtered SAR Data (Downsampled)')
    axes[1].axis('off')

    # Create a custom colormap for the flood map: non-flood (0) as white, flood (1) as blue
    from matplotlib.colors import ListedColormap
    colors = ['white', 'blue'] # 0=white, 1=blue
    cmap_flood = ListedColormap(colors)

    axes[2].imshow(flood_map_downsampled, cmap=cmap_flood, vmin=0, vmax=1)
    axes[2].set_title('Detected Flood Map (Downsampled)')
    axes[2].axis('off')

    # Overlay flood map on original SAR data
    # Create a colormap for overlay: transparent for non-flood, blue for flood
    colors_overlay = ['#FFFFFF00', 'blue'] # Transparent for 0, blue for 1
    cmap_overlay = ListedColormap(colors_overlay)

    axes[3].imshow(original_data_downsampled, cmap='gray', vmin=vmin_original, vmax=vmax_original)
    axes[3].imshow(flood_map_downsampled, cmap=cmap_overlay, vmin=0, vmax=1)
    axes[3].set_title('Flood Overlay (Downsampled)')
    axes[3].axis('off')

    plt.tight_layout()
    plt.show()

def analyze_ratio_histogram(ratio_image):
    """
    Analyzes the histogram of the ratio image to aid in threshold selection.
    Clips the ratio image for better histogram visualization.
    """
    # Clip ratio_image for histogram plotting to a reasonable range
    ratio_image_clipped = np.clip(ratio_image, 0, 5) # Clip to 0-5 for better visualization

    plt.figure(figsize=(8, 6))
    plt.hist(ratio_image_clipped.flatten(), bins=100, color='gray', alpha=0.7)
    plt.title('Histogram of Clipped Ratio Image (0-5)')
    plt.xlabel('Ratio Value')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

def export_geotiff(data, profile, output_filepath):
    """
    Exports a numpy array as a GeoTIFF file.
    """
    # Convert boolean data to uint8 (0 or 1) for GeoTIFF export
    if data.dtype == bool:
        data = data.astype(np.uint8)

    # Update the profile for the output GeoTIFF
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

if __name__ == "__main__":
    # File paths for pre-flood and post-flood SAR images
    pre_flood_tiff_filepath = r"C:\Portfolio\ashburton_flood_sar_analysis\data\S1A_IW_GRDH_1SDV_20210511T173943_20210511T174008_037843_047769_94F3\S1A_IW_GRDH_1SDV_20210511T173943_20210511T174008_037843_047769_94F3.SAFE\measurement\s1a-iw-grd-vv-20210511t173943-20210511t174008-037843-047769-001.tiff"
    post_flood_tiff_filepath = r"C:\Portfolio\ashburton_flood_sar_analysis\data\S1A_IW_GRDH_1SDV_20210604T173944_20210604T174009_038193_0481EC_793C\S1A_IW_GRDH_1SDV_20210604T173944_20210604T174009_038193_0481EC_793C.SAFE\measurement\s1a-iw-grd-vv-20210604t173944-20210604t174009-038193-0481ec-001.tiff"

    print(f"Reading pre-flood SAR TIFF file: {pre_flood_tiff_filepath}")
    pre_flood_sar_data, pre_flood_profile = read_sar_tiff(pre_flood_tiff_filepath)
    print(f"Pre-flood SAR Data Shape: {pre_flood_sar_data.shape}")
    print("Successfully read pre-flood SAR TIFF data.")

    print(f"Reading post-flood SAR TIFF file: {post_flood_tiff_filepath}")
    post_flood_sar_data, post_flood_profile = read_sar_tiff(post_flood_tiff_filepath)
    print(f"Post-flood SAR Data Shape: {post_flood_sar_data.shape}")
    print("Successfully read post-flood SAR TIFF data.")

    # Process pre-flood data
    print("Processing pre-flood data...")
    pre_flood_calibrated_data = radiometric_calibration(pre_flood_sar_data, calibration_factor=0.1)
    pre_flood_filtered_data = apply_speckle_filter(pre_flood_calibrated_data)
    print("Pre-flood data processed successfully.")

    # Process post-flood data
    print("Processing post-flood data...")
    post_flood_calibrated_data = radiometric_calibration(post_flood_sar_data, calibration_factor=0.1)
    post_flood_filtered_data = apply_speckle_filter(post_flood_calibrated_data)
    print("Post-flood data processed successfully.")

    # Downsample data for change detection and histogram analysis to reduce memory usage
    downsample_factor_processing = 4 # Adjust as needed, e.g., 2, 4, 8
    pre_flood_filtered_data_for_processing = pre_flood_filtered_data[::downsample_factor_processing, ::downsample_factor_processing]
    post_flood_filtered_data_for_processing = post_flood_filtered_data[::downsample_factor_processing, ::downsample_factor_processing]

    # Ensure both filtered images have the same shape for change detection
    if pre_flood_filtered_data_for_processing.shape != post_flood_filtered_data_for_processing.shape:
        print(f"Resizing post-flood data from {post_flood_filtered_data_for_processing.shape} to {pre_flood_filtered_data_for_processing.shape} for change detection.")
        post_flood_filtered_data_for_processing = resize(post_flood_filtered_data_for_processing, pre_flood_filtered_data_for_processing.shape, anti_aliasing=True)

    # Ensure no division by zero
    pre_flood_data_safe = np.where(pre_flood_filtered_data_for_processing == 0, 1e-6, pre_flood_filtered_data_for_processing) # Replace 0 with a small number
    ratio_image = post_flood_filtered_data_for_processing / pre_flood_data_safe

    print("Analyzing ratio image histogram...")
    print(f"Ratio Image Min: {np.min(ratio_image):.4f}, Max: {np.max(ratio_image):.4f}, Mean: {np.mean(ratio_image):.4f}, Median: {np.median(ratio_image):.4f}")
    analyze_ratio_histogram(ratio_image)
    print("Histogram analysis complete. Please review the histogram to refine the flood detection threshold.")

    print("Performing change detection...")
    # Clip ratio_image for Otsu's calculation to a reasonable range
    ratio_image_for_otsu = np.clip(ratio_image, 0, 5) # Clip to 0-5 for Otsu's
    # Automatically determined Otsu's threshold (from clipped image): 1.4355
    # otsu_threshold = threshold_otsu(ratio_image_for_otsu)
    # print(f"Automatically determined Otsu's threshold (from clipped image): {otsu_threshold:.4f}")
    ratio_threshold = 0.8 # 手動で調整した閾値

    change_detection_flood_map = perform_change_detection(ratio_image, ratio_threshold)
    print(f"Change Detection Flood Map Shape: {change_detection_flood_map.shape}")
    print(f"Number of flooded pixels (change detection): {np.sum(change_detection_flood_map)}")
    print("Change detection applied successfully.")

    print("Visualizing results...")
    # Visualize the post-flood processed data and the change detection flood map
    visualize_results(post_flood_sar_data, post_flood_filtered_data, change_detection_flood_map)
    print("Visualization complete.")

    # Define path to post-flood annotation XML
    post_flood_xml_filepath = r"C:\Portfolio\ashburton_flood_sar_analysis\data\S1A_IW_GRDH_1SDV_20210604T173944_20210604T174009_038193_0481EC_793C\S1A_IW_GRDH_1SDV_20210604T173944_20210604T174009_038193_0481EC_793C.SAFE\annotation\s1a-iw-grd-vv-20210604t173944-20210604t174009-038193-0481ec-001.xml"

    # Parse XML for georeferencing info
    print(f"Parsing annotation XML: {post_flood_xml_filepath}")
    transform, crs, range_pixel_spacing, azimuth_pixel_spacing = parse_annotation_xml(post_flood_xml_filepath)

    # Update post-flood profile with georeferencing info
    post_flood_profile.update({
        'transform': transform,
        'crs': crs
    })

    # Export flood map as GeoTIFF
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    output_filepath = os.path.join(output_dir, "flood_map_change_detection.tif")
    export_geotiff(change_detection_flood_map, post_flood_profile, output_filepath)
    print("Flood map exported as GeoTIFF.")

    

import os
import warnings
import numpy as np
import rasterio
from rasterio.errors import NotGeoreferencedWarning
from rasterio.windows import Window
from skimage.filters import median
from skimage.morphology import disk
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from rasterio.transform import from_gcps
from rasterio.control import GroundControlPoint
from matplotlib.colors import ListedColormap
from dotenv import load_dotenv
from pathlib import Path
from rasterio.crs import CRS
import pyproj

# 🚫 ジオリファレンス警告を非表示
warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)

# ✅ PROJ データベースの位置を明示
# Point to the PROJ database inside your virtual environment
os.environ['PROJ_LIB'] = pyproj.datadir.get_data_dir()

# -------------------------------------------------------
# Load environment variables and define data paths
# -------------------------------------------------------
load_dotenv()

# 1. Load from .env (values used in Docker)
DATA_DIR = os.getenv("DATA_DIR", "/app/data")

# 2. If using a Windows environment, prioritize the local directory
if os.name == "nt":  # 'nt' is Windows
    win_data_path = "C:/Portfolio/ashburton_flood_sar_analysis/data"
    if os.path.exists(win_data_path):
        DATA_DIR = win_data_path

# Convert to a Path object
DATA_DIR = Path(DATA_DIR)
print(f"[INFO] Using data directory: {DATA_DIR}")

RESULTS_DIR = Path("/app/results")
RESULTS_DIR.mkdir(exist_ok=True)

# -------------------------------------------------------
# ✅ Memory-efficient chunk processing
# -------------------------------------------------------
def process_in_chunks(input_tiff, output_tiff, chunk_size=1024, process_fn=None):
    """
    Processes a large GeoTIFF in chunks to reduce memory usage.
    - input_tiff: input file path
    - output_tiff: output file path
    - chunk_size: tile size (default 1024x1024)
    - process_fn: function applied to each chunk (must return array)
    """
    with rasterio.open(input_tiff) as src:
        profile = src.profile
        profile.update(
            dtype=rasterio.float32,
            compress='lzw',
            transform=src.transform,   # ✅ Keep location information
            crs=src.crs                # ✅ Preserve coordinate reference system
        )

        with rasterio.open(output_tiff, 'w', **profile) as dst:
            width, height = src.width, src.height
            for y in range(0, height, chunk_size):
                for x in range(0, width, chunk_size):
                    w = min(chunk_size, width - x)
                    h = min(chunk_size, height - y)
                    window = Window(x, y, w, h)

                    data = src.read(1, window=window)
                    if process_fn:
                        data = process_fn(data)
                    dst.write(data.astype(np.float32), 1, window=window)

# Example function for speckle noise reduction
def sar_denoise(data):
    """Example SAR processing function"""
    return median(data, disk(3))

# -------------------------------------------------------
# Utility functions
# -------------------------------------------------------
def find_sar_tiff(data_dir, date_str):
    """Finds the first VV SAR file (.tif or .tiff) for a given date."""
    safe_dirs = list(data_dir.glob(f"*{date_str}*.SAFE"))
    if not safe_dirs:
        raise FileNotFoundError(f"No SAFE folder found for {date_str} in {data_dir}")
    
    for safe_dir in safe_dirs:
        meas_dir = safe_dir / "measurement"
        if not meas_dir.exists():
            continue
        files = list(meas_dir.rglob("*vv*.tif*"))
        if files:
            return files[0]
    
    raise FileNotFoundError(f"No VV .tif/.tiff found for {date_str} in {data_dir}")

def find_annotation_xml(data_dir: Path, date_str: str) -> Path:
    """Locate the Sentinel-1 annotation XML file for a given date."""
    for xml_file in data_dir.rglob("*.xml"):
        if date_str in xml_file.name:
            return xml_file
    raise FileNotFoundError(f"No annotation XML found for {date_str}")

def read_sar_tiff(filepath: Path):
    with rasterio.open(filepath) as src:
        return src.read(1), src.profile

def parse_annotation_xml(xml_filepath: Path):
    tree = ET.parse(xml_filepath)
    root = tree.getroot()
    image_info = root.find('.//imageInformation')
    range_pixel_spacing = float(image_info.find('rangePixelSpacing').text)
    azimuth_pixel_spacing = float(image_info.find('azimuthPixelSpacing').text)
    gcps = [
        GroundControlPoint(
            row=int(pt.find('line').text),
            col=int(pt.find('pixel').text),
            x=float(pt.find('longitude').text),
            y=float(pt.find('latitude').text)
        )
        for pt in root.findall('.//geolocationGridPoint')
    ]
    transform = from_gcps(gcps)
    crs = 'EPSG:4326'
    return transform, crs, range_pixel_spacing, azimuth_pixel_spacing

def radiometric_calibration(sar_data, factor=1.0):
    return sar_data * factor

def perform_change_detection(ratio_image, threshold_ratio=0.8):
    return ratio_image < threshold_ratio

def analyze_ratio_histogram(ratio_image):
    ratio_clipped = np.clip(ratio_image, 0, 5)
    plt.figure(figsize=(8, 6))
    plt.hist(ratio_clipped.flatten(), bins=100, color='gray', alpha=0.7)
    plt.title("Histogram of Ratio Image (0–5)")
    plt.xlabel("Ratio")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.savefig(RESULTS_DIR / "histogram.png")
    plt.close()
    print("✅ Histogram saved")

def visualize_results(original, filtered, flood_map, downsample=10):
    orig_ds = original[::downsample, ::downsample]
    filt_ds = filtered[::downsample, ::downsample]
    flood_ds = flood_map[::downsample, ::downsample]

    vmin_o, vmax_o = np.percentile(orig_ds, [2, 98])
    vmin_f, vmax_f = np.percentile(filt_ds, [2, 98])

    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    axes[0].imshow(orig_ds, cmap='gray', vmin=vmin_o, vmax=vmax_o)
    axes[0].set_title('Original SAR'); axes[0].axis('off')
    axes[1].imshow(filt_ds, cmap='gray', vmin=vmin_f, vmax=vmax_f)
    axes[1].set_title('Filtered SAR'); axes[1].axis('off')
    axes[2].imshow(flood_ds, cmap=ListedColormap(['white', 'blue']), vmin=0, vmax=1)
    axes[2].set_title('Flood Map'); axes[2].axis('off')
    axes[3].imshow(orig_ds, cmap='gray', vmin=vmin_o, vmax=vmax_o)
    axes[3].imshow(flood_ds, cmap=ListedColormap(['#FFFFFF00','blue']), vmin=0, vmax=1)
    axes[3].set_title('Flood Overlay'); axes[3].axis('off')

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "Figure_1.png")
    plt.close()
    print("✅ Figure saved")

def export_geotiff(data, profile, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if data.dtype == bool:
        data = data.astype(np.uint8)

    out_profile = profile.copy()
    out_profile.update({
        'dtype': data.dtype,
        'height': data.shape[0],
        'width': data.shape[1],
        'count': 1,
        'compress': 'lzw',
        'crs': profile.get('crs') or CRS.from_epsg(4326),  # ✅ fallback CRS
        'transform': profile.get('transform') or rasterio.transform.from_origin(0, 0, 1, 1)  # ✅ fallback transform
    })

    with rasterio.open(output_path, 'w', **out_profile) as dst:
        dst.write(data, 1)
    print(f"✅ GeoTIFF exported: {output_path}")

# -------------------------------------------------------
# Main
# -------------------------------------------------------
if __name__ == "__main__":
    pre_flood_file = find_sar_tiff(DATA_DIR, "20210511")
    post_flood_file = find_sar_tiff(DATA_DIR, "20210604")
    post_xml_file = find_annotation_xml(DATA_DIR, "20210604")

    # ✅ Chunk-based denoising
    pre_filtered_tiff = RESULTS_DIR / "pre_flood_filtered.tif"
    post_filtered_tiff = RESULTS_DIR / "post_flood_filtered.tif"

    process_in_chunks(pre_flood_file, pre_filtered_tiff, chunk_size=1024, process_fn=sar_denoise)
    process_in_chunks(post_flood_file, post_filtered_tiff, chunk_size=1024, process_fn=sar_denoise)

    # ✅ Load filtered data
    pre_data, pre_profile = read_sar_tiff(pre_filtered_tiff)
    post_data, post_profile = read_sar_tiff(post_filtered_tiff)

    # Crop and analyze
    min_rows = min(pre_data.shape[0], post_data.shape[0])
    min_cols = min(pre_data.shape[1], post_data.shape[1])
    pre_cropped = pre_data[:min_rows, :min_cols]
    post_cropped = post_data[:min_rows, :min_cols]

    ratio = post_cropped / np.where(pre_cropped == 0, 1e-6, pre_cropped)
    analyze_ratio_histogram(ratio)
    flood_map = perform_change_detection(ratio)
    visualize_results(post_data, post_data, flood_map)

    transform, crs, _, _ = parse_annotation_xml(post_xml_file)

    # ✅ CRS設定を安全に実行（proj.db がなくても fallback）
    try:
        post_profile.update({
            'transform': transform,
            'crs': CRS.from_epsg(4326)
        })
    except Exception as e:
        print(f"[WARN] Could not load EPSG:4326 from PROJ database: {e}")
        # 手動でWGS84を指定（proj.db不要）
        post_profile.update({
            'transform': transform,
            'crs': {'init': 'EPSG:4326'}
        })

    # ✅ GeoTIFF出力（ここは変更不要）
    export_geotiff(flood_map, post_profile, RESULTS_DIR / "flood_map_change_detection.tif")


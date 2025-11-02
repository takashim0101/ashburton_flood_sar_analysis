import rasterio
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, jaccard_score, f1_score, classification_report
from pathlib import Path
import os
from dotenv import load_dotenv

def compare_flood_maps(change_detect_path, ml_predict_path, output_dir):
    """Compares two flood map GeoTIFFs quantitatively and visually."""
    try:
        with rasterio.open(change_detect_path) as src1:
            change_map = src1.read(1)
        
        with rasterio.open(ml_predict_path) as src2:
            ml_map = src2.read(1)

    except rasterio.errors.RasterioIOError as e:
        print(f"Error reading GeoTIFF files: {e}")
        print("Please ensure both process_sar.py and predict_ml_flood.py have been run successfully.")
        return

    # Ensure maps are the same shape
    print(f"Shape of Traditional Map ('{change_detect_path.name}'): {change_map.shape}")
    print(f"Shape of ML Map ('{ml_predict_path.name}'): {ml_map.shape}")

    if change_map.shape != ml_map.shape:
        print("\nError: Input maps have different shapes. Cannot compare.")
        return

    # Flatten arrays for metrics calculation
    change_flat = change_map.flatten()
    ml_flat = ml_map.flatten()

    # --- Quantitative Comparison ---
    print("--- Quantitative Comparison Report ---")
    print(f"Comparing '{change_detect_path.name}' (Traditional) vs. '{ml_predict_path.name}' (ML)")

    # Jaccard Index (IoU)
    j_score = jaccard_score(change_flat, ml_flat, average='binary')
    print(f"\nJaccard Score (IoU): {j_score:.4f}")
    print("(Measures the intersection over the union of the two maps. Higher is better.)")

    # Dice Coefficient (F1 Score)
    f1 = f1_score(change_flat, ml_flat, average='binary')
    print(f"\nDice Coefficient (F1 Score): {f1:.4f}")
    print("(Another measure of overlap, sensitive to class balance. Higher is better.)")

    # Detailed Classification Report
    print("\nClassification Report (ML map vs. Traditional map as reference):")
    # target_names: 0 = Non-Flood, 1 = Flood
    report = classification_report(change_flat, ml_flat, target_names=['Non-Flood', 'Flood'])
    print(report)

    # --- Confusion Matrix Visualization ---
    cm = confusion_matrix(change_flat, ml_flat)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Predicted Non-Flood', 'Predicted Flood'],
                yticklabels=['Actual Non-Flood', 'Actual Flood'])
    plt.title('Confusion Matrix: ML Prediction vs. Traditional Change Detection')
    plt.xlabel('ML (U-Net) Prediction')
    plt.ylabel('Traditional Change Detection')
    
    # Save the confusion matrix plot
    cm_path = output_dir / "Figure_Confusion_Matrix.png"
    plt.savefig(cm_path)
    print(f"\nConfusion matrix plot saved to: {cm_path}")
    plt.close()

if __name__ == "__main__":
    load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / '.env')

    RESULTS_DIR = Path(os.getenv("RESULTS_DIR", "results"))
    RESULTS_DIR.mkdir(exist_ok=True)

    # Define paths to the two GeoTIFFs to compare
    change_detection_tiff = RESULTS_DIR / "flood_map_change_detection.tif"
    ml_prediction_tiff = RESULTS_DIR / "flood_map_unet_prediction.tif"

    compare_flood_maps(change_detection_tiff, ml_prediction_tiff, RESULTS_DIR)
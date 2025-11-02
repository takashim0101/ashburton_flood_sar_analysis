import geopandas
import os

fgdb_path = "/c/Portfolio/ashburton_flood_sar_analysis/data/statsnz-territorial-authority-2025/territorial-authority-2025-clipped.gdb"

try:
    # List layers in the FGDB
    layers = geopandas.io.file.fiona.listlayers(fgdb_path)
    print(f"Available layers in FGDB: {layers}")

    # Assuming the main layer is the first one, or named appropriately
    # Let's try to read the first layer
    if layers:
        layer_name = layers[0]
        gdf = geopandas.read_file(fgdb_path, layer=layer_name)
        print(f"Columns in layer '{layer_name}': {gdf.columns.tolist()}")
        
        # Try to find Ashburton District
        # Common column names for administrative units are 'Name', 'TA_NAME', 'DISTRICT', etc.
        # Filter for Ashburton District using the provided ID
        ashburton_gdf = gdf[gdf['TA2025_V1_00'] == '063']
        
        if not ashburton_gdf.empty:
            print("Ashburton District GeoDataFrame:")
            print(ashburton_gdf)
            
            # Extract WKT and bounding box
            ashburton_geometry = ashburton_gdf.geometry.iloc[0]
            ashburton_wkt = ashburton_geometry.wkt
            ashburton_bounds = ashburton_geometry.bounds # (minx, miny, maxx, maxy)
            
            print(f"\nAshburton WKT: {ashburton_wkt}")
            print(f"Ashburton Bounds (min_lon, min_lat, max_lon, max_lat): {ashburton_bounds}")
        else:
            print("Ashburton District not found in any common name column.")
    else:
        print("No layers found in the FGDB.")

except Exception as e:
    print(f"Error reading FGDB or processing data: {e}")
    print("Please ensure geopandas and its dependencies (like fiona, GDAL) are installed and configured correctly.")
    print("You might need to install geopandas: pip install geopandas")

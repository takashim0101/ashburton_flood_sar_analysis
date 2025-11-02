import os
import psycopg2
from dotenv import load_dotenv
from urllib.parse import quote_plus

# ------------------------------------------------------------
#  Load environment variables
# ------------------------------------------------------------
load_dotenv()

# --- Database configuration ---
DB_HOST = "127.0.0.1"
DB_NAME = os.getenv("POSTGRES_DB", "ashburton_db")
DB_USER = os.getenv("POSTGRES_USER", "docker_user")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD", "P@ssw0rd123!")
DB_PORT = "5432"

# --- Target table name ---
TABLE_NAME = "flood_map"

print(f"Attempting guaranteed import into PostGIS container: {DB_NAME}...")

# ------------------------------------------------------------
#  Step 1: Build the raster2pgsql import command
# ------------------------------------------------------------
# Note: Using absolute paths inside the PostGIS container.
#       The /data and /results folders must be mounted in docker-compose.yml.
RASTER2PGSQL_PATH = "/usr/bin/raster2pgsql"  # update after you check with 'which'
docker_command = (
    f'docker exec -e PGPASSWORD="{DB_PASSWORD}" ashburton_postgis '
    f'sh -c "{RASTER2PGSQL_PATH} -C -s 4326 -I -M '
    f'/data/results/flood_map_change_detection.tif {TABLE_NAME} | '
    f'psql -U {DB_USER} -d {DB_NAME} -h localhost -p 5432"'
)

# ------------------------------------------------------------
#  Step 2: Construct the docker exec command
# ------------------------------------------------------------
# - Use double quotes for PowerShell compatibility.
# - Use absolute PostgreSQL binary paths to avoid PATH issues.
docker_command = (
    f'docker exec -e PGPASSWORD="{DB_PASSWORD}" ashburton_postgis '
    f'sh -c "{RASTER2PGSQL_COMMAND} | '
    f'/usr/lib/postgresql/14/bin/psql -U {DB_USER} -d {DB_NAME} -h localhost -p 5432"'
)

print("\nExecuting guaranteed import via Docker exec...\n")
print(f"Running command:\n{docker_command}\n")

# ------------------------------------------------------------
#  Step 3: Run the import and handle results
# ------------------------------------------------------------
try:
    import_result = os.popen(docker_command).read()

    if "Raster column added" in import_result or "COMMIT" in import_result:
        print("🎉 ✅ Raster data successfully imported into PostGIS!")
        print(f"Table name: {TABLE_NAME}")

        # --------------------------------------------------------
        #  Step 4: Update SRID to EPSG:4326 using psycopg2
        # --------------------------------------------------------
        try:
            encoded_password = quote_plus(DB_PASSWORD)
            conn_string = f"postgresql://{DB_USER}:{encoded_password}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
            conn = psycopg2.connect(conn_string)
            cursor = conn.cursor()

            update_sql = f"UPDATE {TABLE_NAME} SET rast = ST_SetSRID(rast, 4326);"
            cursor.execute(update_sql)
            conn.commit()
            conn.close()
            print("✅ SRID updated to EPSG:4326 successfully.")

        except psycopg2.Error as e_update:
            print(f"⚠️ Warning: Data imported but SRID update failed. Error: {e_update}")

    else:
        print("❌ Data import failed. Please check Docker logs or file paths.")
        print(f"\n--- Docker Output ---\n{import_result}")

except Exception as e_main:
    print(f"❌ Fatal error during execution: {e_main}")

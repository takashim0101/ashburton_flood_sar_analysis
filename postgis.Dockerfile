# Extend the official PostGIS image
FROM postgis/postgis:14-3.4

# Install raster2pgsql and related PostgreSQL utilities
RUN apt-get update && apt-get install -y postgis postgresql-client && rm -rf /var/lib/apt/lists/*

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as cx
from shapely.geometry import Point

gauges_df = pd.read_csv('usgs_vel_site_locations.csv')

geoms = [Point(lon, lat) for lat, lon in gauges_df[['lat_gauge', 'lon_gauge']].values]
gauges_gdf = gpd.GeoDataFrame(gauges_df, geometry=geoms)
gauges_gdf.set_crs('EPSG:4326', inplace=True)
# Set up the figure
fig, ax = plt.subplots(figsize=(6, 6))

# Plot the geometries from the GeoDataFrame
gauges_gdf.plot(ax=ax, color='blue', markersize=20)

# Add latitude and longitude labels
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
plt.ylim(0, 75)

# Add title
ax.set_title('USGS Gauges With Velocity Data')

# Customize the appearance of the plot
ax.grid(True, linestyle='--', linewidth=0.5, color='gray')  # Add grid lines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

cx.add_basemap(ax, crs=gauges_gdf.crs.to_string(), source=cx.providers.Stamen.TonerLite)
plt.rcParams['interactive'] = False
# Add a legend if needed
# ax.legend(['Legend Label'])

# Show the plot
plt.tight_layout()
plt.savefig('usgs_gauges_map.png', format='png', dpi=400, bbox_inches='tight')

plt.show()

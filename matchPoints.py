import math
import time
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point

old_streams = gpd.read_file('central_america-geoglows-drainageline.gpkg')
new_streams = gpd.read_file('TDX_streamnet_7020065090_01.gpkg')
gauge_filters = pd.read_csv('gauge_table_filters.csv')
old_streams = old_streams.merge(gauge_filters[['model_id', 'gauge_id']],
                                how='left', left_on='COMID', right_on='model_id')

new_streams = new_streams.to_crs(old_streams.crs)

buffer_distance = 50

old_streams['nga_id'] = ''
old_cpy = old_streams

start_time = time.time()

for idx, old_stream in old_streams.loc[~old_streams['gauge_id'].isna()].iterrows():
    old_geom = old_stream.geometry
    if old_geom.type == 'MultiLineString':
        old_geom = old_geom.geoms[0]
    gauge_point = Point(
        gauge_filters.loc[
            gauge_filters['model_id'] == old_stream['COMID'], 'lon_gauge'
        ].values[0],
        gauge_filters.loc[
            gauge_filters['model_id'] == old_stream['COMID'], 'lat_gauge'
        ].values[0],
    )
    data = {'geometry': [gauge_point]}
    gauge_geom = gpd.GeoDataFrame(data, crs='EPSG:4326') \
        .to_crs(old_streams.crs).geometry

    buffer_riv = old_geom.buffer(buffer_distance)
    buffer_gauge = gauge_geom.buffer(buffer_distance)

    intersecting_new_streams = new_streams[new_streams.intersects(buffer_riv)]
    # intersecting_new_streams = intersecting_new_streams[
    #     intersecting_new_streams.intersects(buffer_gauge)]
    if intersecting_new_streams.empty:
        continue
    order_counts = intersecting_new_streams.groupby('strmOrder').agg(
        Count=('strmOrder', 'size')).sort_values('Count', ascending=False).reset_index()
    if not order_counts.empty:
        max_order = order_counts.loc[order_counts['strmOrder'] != 1, 'strmOrder'].iloc[0]
        intersecting_new_streams = intersecting_new_streams.loc[intersecting_new_streams['strmOrder'] == max_order]
    closest_stream = None
    closest_distance = float('inf')
    angle_threshold = 20

    # Iterate over the intersecting higher-resolution rivers
    for i, new_stream in intersecting_new_streams.iterrows():
        new_geom = new_stream.geometry

        # Calculate the distance between the lower and higher stream
        distance = gauge_geom.distance(new_geom).min()
        # distance = old_geom.distance(new_geom)

        # Check if the current higher stream has a closer distance and similar direction
        if distance < closest_distance:
            print(distance)
            # Calculate the angle difference between the lower and higher stream
            lower_coords = np.array(old_geom.coords)
            higher_coords = np.array(new_geom.coords)

            # Calculate the vectors
            lower_vector = lower_coords[-1] - lower_coords[0]
            higher_vector = higher_coords[-1] - higher_coords[0]

            # Calculate dot product and magnitude product
            dot_product = np.dot(lower_vector, higher_vector)
            magnitude_product = np.linalg.norm(lower_vector) * np.linalg.norm(higher_vector)

            # Calculate the angle difference
            angle_difference = math.acos(dot_product / magnitude_product)

            # Convert angle difference to degrees
            angle_difference_degrees = 180 - math.degrees(angle_difference)

            # Adjust the angle threshold as needed
            if angle_difference_degrees < angle_threshold:
                closest_stream = new_stream
                closest_distance = distance

    # Process the closest_stream if it exists
    if closest_stream is not None:
        old_cpy.loc[old_cpy.index[idx], 'nga_id'] = str(closest_stream['LINKNO'])

# End the timer
end_time = time.time()

# Calculate the elapsed time
elapsed_time = end_time - start_time

# Print the elapsed time
print(f"Elapsed time: {elapsed_time} seconds")
filtered_assigned = old_cpy.loc[old_cpy['nga_id'] != '']
filtered_assigned['nga_id'] = filtered_assigned['nga_id'].astype(int)

gpd.GeoDataFrame(filtered_assigned).to_file('assigned_gauges.gpkg')

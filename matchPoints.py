# from fuzzywuzzy import fuzz
import math

import geopandas as gpd
import pandas as pd
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
for idx, old_stream in old_streams.loc[~old_streams['gauge_id'].isna()].iterrows():
    old_geom = old_stream.geometry
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
    # todo Make pivot table with number of occurrences of each stream order, use the one with maximum occurrences to
    # todo determine which stream order will match the old river. There may be many order 1's so those should be
    # todo filtered out unless there's only 1 or 2? Lots of conditional thinking needs to go into this.
    order_counts = intersecting_new_streams.groupby('strmOrder').agg(
        Count=('strmOrder', 'size')).reset_index().sort_values('Count', ascending=False)
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
            # Calculate the angle difference between the lower and higher stream
            lower_vector = old_geom.coords[-1][0] - old_geom.coords[0][0], old_geom.coords[-1][1] - \
                           old_geom.coords[0][1]
            higher_vector = new_geom.coords[-1][0] - new_geom.coords[0][0], new_geom.coords[-1][
                1] - new_geom.coords[0][1]
            dot_product = lower_vector[0] * higher_vector[0] + lower_vector[1] * higher_vector[1]
            magnitude_product = math.sqrt(lower_vector[0] ** 2 + lower_vector[1] ** 2) * math.sqrt(
                higher_vector[0] ** 2 + higher_vector[1] ** 2)
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
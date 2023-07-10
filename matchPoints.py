import os
import time

import geopandas as gpd
import math
import numpy as np
import pandas as pd
from shapely.geometry import Point


def match_new_to_old_rivers(old_drain,
                            new_drain,
                            old_matched,
                            buffer_distance: int = 50,
                            old_strmid: str = 'COMID',
                            new_strmid: str = 'LINKNO',
                            gauge_strmid: str = 'model_id',
                            gauge_siteid: str = 'gauge_id',
                            new_strmord: str = 'strmOrder',
                            old_us_area: str = 'Tot_Drain_',
                            new_us_area: str = 'DSContArea',
                            lat_col: str = 'lat_gauge',
                            lon_col: str = 'lon_gauge',
                            ):
    old_streams = gpd.read_file(old_drain)
    new_streams = gpd.read_file(new_drain)
    gauge_filters = pd.read_csv(old_matched)
    old_streams = old_streams.merge(gauge_filters[[gauge_strmid, gauge_siteid]],
                                    how='inner', left_on=old_strmid, right_on=gauge_strmid)

    new_streams = new_streams.to_crs(old_streams.crs)

    old_streams['nga_id'] = ''
    old_streams['area_ratio'] = -1.
    old_cpy = old_streams

    start_time = time.time()

    # for idx, old_stream in old_streams.loc[~old_streams[gauge_siteid].isna()].iterrows():
    for idx, old_stream in old_streams.iterrows():
        old_geom = old_stream.geometry
        if old_geom.geom_type == 'MultiLineString':
            old_geom = old_geom.geoms[0]
        gauge_point = Point(
            gauge_filters.loc[
                gauge_filters[gauge_strmid] == old_stream[old_strmid], lon_col
            ].values[0],
            gauge_filters.loc[
                gauge_filters[gauge_strmid] == old_stream[old_strmid], lat_col
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
        order_counts = intersecting_new_streams.groupby(new_strmord).agg(
            Count=(new_strmord, 'size')).sort_values('Count', ascending=False).reset_index()
        if not order_counts.empty:
            max_order = order_counts.loc[order_counts[new_strmord] != 1, new_strmord].iloc[0]
            intersecting_new_streams = intersecting_new_streams.loc[intersecting_new_streams[new_strmord] == max_order]
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
            old_cpy.loc[old_cpy.index[idx], 'nga_id'] = str(closest_stream[new_strmid])
            old_cpy.loc[old_cpy.index[idx], 'area_ratio'] = \
                closest_stream[new_us_area] / old_cpy.loc[old_cpy.index[idx], old_us_area]


    # End the timer
    end_time = time.time()

    # Calculate the elapsed time
    elapsed_time = end_time - start_time

    # Print the elapsed time
    print(f"Elapsed time: {elapsed_time} seconds")
    filtered_assigned_old = old_cpy.loc[old_cpy['nga_id'] != ''].reset_index(drop=True)
    for col in [gauge_strmid, 'nga_id']:
        filtered_assigned_old.loc[:, col] = filtered_assigned_old.loc[:, col].astype(int)

    gauge_assignments = gauge_filters[[gauge_siteid, gauge_strmid, lat_col, lon_col]]\
        .merge(filtered_assigned_old[[gauge_siteid, 'nga_id', 'area_ratio', 'geometry']], on=gauge_siteid, how='left')[
        [gauge_siteid, gauge_strmid, 'nga_id', lat_col, lon_col, 'area_ratio', 'geometry']
    ].rename(columns={'geometry': 'geoglows_geom', gauge_strmid: 'geoglows_id'})

    gauge_assignments = gauge_assignments.merge(
        new_streams[[new_strmid, 'geometry']], left_on='nga_id', right_on=new_strmid).drop(new_strmid, axis=1)\
        .rename(columns={'geometry': 'nga_geom'})

    for col in ['geoglows_id', 'nga_id']:
        gauge_assignments[col] = gauge_assignments[col].astype(int)
    gauge_assignments.to_csv('gauge_assignments_extraattrs.csv')
    gauge_lookup = gauge_assignments[[gauge_siteid, 'geoglows_id', 'nga_id']]
    gauge_lookup.to_csv('gauge_lookup.csv')
    return gauge_assignments

if __name__ == "__main__":
    geoglows_drain = 'central_america-geoglows-drainageline.gpkg'
    nga_drain = 'TDX_streamnet_7020065090_01.gpkg'
    matched_geoglows = 'gauge_table_filters.csv'
    match_new_to_old_rivers(geoglows_drain, nga_drain, matched_geoglows)


import math
import os
import time
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

warnings.filterwarnings('ignore')


def match_new_to_old_rivers(old_streams,
                            new_streams,
                            gauge_filters,
                            out_dir: str = None,
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
                            filter_strmord: bool = True
                            ):
    """


    Args:
        old_streams (DataFrame-like): DataFrame of GEOGloWS streams.
        new_streams (DataFrame-like): DataFrame of NGA streams.
        gauge_filters (DataFrame-like): DataFrame with gauge ids, assigned GEOGloWS ids, and lat lon of gauge locations.
        out_dir (str, optional): Directory to write outputs. If None, won't write. Defaults to None.
        buffer_distance (int, optional): Directly fed into buffer function. Defaults to 50.
        old_strmid (str, optional): GEOGloWS unique river id. Defaults to 'COMID'.
        new_strmid (str, optional): NGA unique river id. Defaults to 'LINKNO'.
        gauge_strmid (str, optional): Name in old_matched for GEOGloWS id. Defaults to 'model_id'.
        gauge_siteid (str, optional): Name for gauge id. Defaults to 'gauge_id'.
        new_strmord (str, optional): Name for stream order column in new_drain. Defaults to 'strmOrder'.
        old_us_area (str, optional): Name for upstream area in old_drain. Defaults to 'Tot_Drain_'.
        new_us_area (str, optional): Name for upstream area in new_drain. Defaults to 'DSContArea'.
        lat_col (str, optional): Name for latitude column in old_matched. Defaults to 'lat_gauge'.
        lon_col (str, optional): Name for longitude column in old_matched. Defaults to 'lon_gauge'.
        filter_strmord (bool, optional): Determines whether to filter by stream order. Set to False
                                         if no stream order column is available in new_streams. Defaults to True.

    Returns:
        pd.DataFrame: Dataframe with GEOGloWS river, matched NGA river, matched gauge, ratio of NGA area / GEOGloWS,
                      and geometries for all three pieces
    """
    # old_streams = gpd.read_file(old_drain)
    # new_streams = gpd.read_file(new_drain)
    # gauge_filters = pd.read_csv(old_matched)
    old_streams = old_streams.merge(gauge_filters[[gauge_strmid, gauge_siteid]],
                                    how='inner', left_on=old_strmid, right_on=gauge_strmid)
    if old_streams.empty:
        print('No gauges assigned to region')
        return None

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
        if filter_strmord:
            order_counts = intersecting_new_streams.groupby(new_strmord).agg(
                Count=(new_strmord, 'size')).sort_values('Count', ascending=False).reset_index()
            if not order_counts.empty:
                max_order = order_counts.loc[order_counts[new_strmord] != 1, new_strmord].iloc[0]
                intersecting_new_streams = intersecting_new_streams.loc[intersecting_new_streams[new_strmord] == max_order]
        closest_stream = None
        closest_distance = float('inf')
        angle_threshold = 10

        # Iterate over the intersecting higher-resolution rivers
        check_df = pd.DataFrame(columns=['river', 'distance', 'angle'])
        if not intersecting_new_streams.empty:
            for i, new_stream in intersecting_new_streams.iterrows():
                new_geom = new_stream.geometry
                if new_geom.geom_type == 'MultiLineString':
                    new_geom = new_geom.geoms[0]  # may not be enough

                # Calculate the distance between the lower and higher stream
                distance = gauge_geom.distance(new_geom).min()
                # distance = old_geom.distance(new_geom)

                # Check if the current higher stream has a closer distance and similar direction
                # if distance < closest_distance:
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
                if check_df.empty:
                    check_df = pd.DataFrame(data={
                                   'river': new_stream[new_strmid],
                                   'distance': distance,
                                   'angle': angle_difference_degrees
                               }, index=[i])
                    continue
                check_df = pd.concat([check_df,
                                      pd.DataFrame({
                                          'river': new_stream[new_strmid],
                                          'distance': distance,
                                          'angle': angle_difference_degrees
                                      }, index=[i])], axis=0)


                # Adjust the angle threshold as needed
                # if angle_difference_degrees < angle_threshold:
                #         closest_stream = new_stream
                #         closest_distance = distance
            closest_distance = check_df['distance'].min()
            min_angle = check_df['angle'].min()
            closest_stream = intersecting_new_streams.loc[intersecting_new_streams[new_strmid] == check_df.loc[
                        check_df['distance'] == closest_distance, 'river'].values[0]]
            while True:
                if not check_df.empty and ((check_df.loc[
                         check_df['distance'] == closest_distance, 'angle'] - min_angle) > angle_threshold).values[0]:
                    if check_df.shape[0] <= 1:
                        break
                    check_df = check_df.loc[check_df['distance'] != closest_distance]
                    closest_distance = check_df['distance'].min()
                    if not check_df.loc[check_df['distance'] == closest_distance, 'river'].empty:
                        try:
                            closest_stream = intersecting_new_streams.loc[
                                intersecting_new_streams[new_strmid] == check_df.loc[
                                    check_df['distance'] == closest_distance, 'river'].values[0]]
                        except:
                            continue
                    continue
                break

        # Process the closest_stream if it exists
        if closest_stream is not None:
            old_cpy.loc[old_cpy.index[idx], 'nga_id'] = closest_stream[new_strmid].astype(str).values[0]
            old_cpy.loc[old_cpy.index[idx], 'area_ratio'] = \
                (closest_stream.loc[:, new_us_area] / old_cpy.loc[old_cpy.index[idx], old_us_area]).values[0]


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

    if out_dir:
        os.chdir(out_dir)
        gauge_assignments.to_csv('gauge_assignments_extraattrs.csv')
        gauge_lookup = gauge_assignments[[gauge_siteid, 'geoglows_id', 'nga_id']]
        gauge_lookup.to_csv('gauge_lookup.csv')
    return gauge_assignments


def match_rivers_to_gauges(drain: gpd.GeoDataFrame, new_gauges: pd.DataFrame,
                           new_strm_id = 'TDXHydroLinkNo',
                           new_strm_ord = 'strmOrder',
                           magnitude_col = 'Magnitude',
                           dist_wt = 1,
                           mag_wt = 0,
                           lat_col = 'lat_gauge',
                           lon_col = 'lon_gauge',
                           ) -> pd.DataFrame:
    """
    Matches rivers directly to gauges by scoring rivers in the vicinity of a gauge based
    on distance, number of other rivers of common stream order in that vicinity, and magnitude
    (though that is weighted zero by default)

    Args:
        drain (gpd.GeoDataFramev): Drainage network
        new_gauges (pd.DataFrame): Gauges with an id, lat, and lon column (though id isn't directly used)
        new_strm_id (str, optional): Column name for the stream ID from drain. Defaults to 'TDXHydroLinkNo'.
        new_strm_ord (str, optional): Column name for the stream order from drain. Defaults to 'strmOrder'.
        magnitude_col (str, optional): Column name for the magnitude (Shreve stream order). Defaults to 'Magnitude'.
        dist_wt (int, optional): Amount to emphasize inverse distance of river to gauge for scoring.
                                 Should combine with mag_wt to add up to 1. Defaults to 1.
        mag_wt (int, optional): Amount to emphasize magnitude of river in scoring. Defaults to 0.
        lat_col (str, optional): Name of latitude column from new_gauges. Defaults to 'lat_gauge'.
        lon_col (str, optional): Name of longitude column from new_gauges. Defaults to 'lon_gauge'.

    Returns:
        pd.DataFrame: A version of the new_gauges dataframe with an nga_id column that contains the
                      assigned IDs from drain.
    """
    new_gauges['nga_id'] = ''
    if magnitude_col is None:
        drain['Magnitude'] = 0
        magnitude_col = 'Magnitude'
    for i in range(len(new_gauges)):
        gauge = new_gauges.iloc[i]
        gauge_geom = Point(gauge[lon_col], gauge[lat_col])
        drain['distance'] = drain.distance(gauge_geom)
        nearest_rivs = drain.loc[drain[new_strm_ord] != 1].sort_values(by='distance').head(10)
        order_counts = nearest_rivs.groupby(new_strm_ord).agg(
            Count=(new_strm_ord, 'size')).sort_values('Count', ascending=False).reset_index()
        gauge_geom = Point(gauge[lon_col], gauge[lat_col])

        # Buffer streams to narrow down distances to be calculated
        gauge_buffer = gpd.GeoDataFrame({'geometry': [gauge_geom]}, crs='EPSG:4326').geometry.buffer(0.25)
        drain_filtered = gpd.sjoin(drain, gauge_buffer.to_frame(), how='inner', op='intersects')[drain.columns]

        if drain_filtered.empty:
            print(f'Gauge of index {gauge.name} not close enough to any rivers')
            new_gauges.loc[new_gauges.index[i], 'nga_id'] = '-1'
            continue

        # Calculate distances, get 10 closest
        drain_filtered['distance'] = drain_filtered.distance(gauge_geom)
        nearest_rivs = drain_filtered.loc[drain_filtered[new_strm_ord] != 1].sort_values(by='distance').head(10)
        if nearest_rivs.empty:
            # Set to empty value if not close enough to any
            new_gauges.loc[new_gauges.index[i], 'nga_id'] = '-1'
            continue

        # Count rivers of a given stream order, score by distance and prevalence of stream order among other rivers
        order_counts = nearest_rivs.groupby(new_strm_ord).agg(
            Count=(new_strm_ord, 'size')).sort_values('Count', ascending=False).reset_index()
        if not order_counts.empty:
            max_order = order_counts.loc[order_counts[new_strm_ord] != 1, new_strm_ord].iloc[0]
        nearest_rivs['score'] = nearest_rivs[new_strm_ord].apply(
            lambda x: order_counts.loc[order_counts[new_strm_ord] == x, 'Count'].values[0]) + \
            nearest_rivs[magnitude_col] * mag_wt + dist_wt / nearest_rivs['distance']
        nearest_riv = nearest_rivs.loc[nearest_rivs['score'] == nearest_rivs['score'].max()]

        # new_gauges.loc[new_gauges.index[i], 'nga_id'] = nearest_riv[new_strm_id]
        # if (nearest_riv[new_strm_id] != gauge['nga_id']).iloc[0]:
        new_gauges.loc[new_gauges.index[i], 'nga_id'] = str(nearest_riv[new_strm_id].values[0])
        # if (nearest_riv[new_strmid] != gauge['nga_id']).iloc[0]:
        #     changed_count += 1
        #     gauge['nga_id'] = nearest_riv[new_strm_id].values[0]
        #     gauge = gauge.to_frame().T
        #     if changed_df.empty:
        #         changed_df = gauge
        #     else:
        #         changed_df = pd.concat([changed_df, gauge], axis=0).reset_index(drop=True)
    new_gauges['nga_id'] = new_gauges['nga_id'].apply(lambda x: x.split('.')[0])
    return new_gauges


corrected_dr_gaugestbl = 'gauge_assignments/corrected_dr_gauges.csv'

if __name__ == "__main__":
    geoglows_drain = 'central_america-geoglows-drainageline.gpkg'
    nga_drain = 'TDX_streamnet_7020065090_01.gpkg'
    nga_drain_dr = 'tdxhydro/vpu_700/vpu_718_streams.gpkg'
    matched_geoglows = 'gauge_table_filters.csv'
    # old_streams = gpd.read_file(geoglows_drain)
    gauge_assignments = pd.read_csv(corrected_dr_gaugestbl, index_col=0)
    dr_streams = gpd.read_file(nga_drain_dr)
    # gauge_filters = pd.read_csv(matched_geoglows)
    # match_new_to_old_rivers(old_streams, new_streams, gauge_filters, 'gauge_assignments')
    match_rivers_to_gauges(dr_streams, gauge_assignments)

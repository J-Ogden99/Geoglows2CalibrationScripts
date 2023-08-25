import os
from glob import glob
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

from geoglows.streamflow import reach_to_region
import HydroErr.HydroErr as Hs

V1_DIR = '/Users/joshogden/Downloads/'
V2_DIR = '/Users/joshogden/Downloads/'
V2_LOOKUP_TBL = 'master_table.parquet'
GAUGE_DIR = '/Users/joshogden/Downloads/ObservedDischarge-selected'

# Must match function names in HydroErr/Hydrostats
STATS_LIST = ['me', 'rmse', 'kge_2012', 'r_squared', 'nse', 'mape']
# Used to rename stat columns
STATS_NAMES = ['Mean Error', 'Root Mean Squared Error', 'Kling-Gupta Efficiency', 'R^2', 'Nash-Sutcliffe Efficiency',
               'Mean Absolute Percent Error']


def calc_error_metrics(gauge_data: pd.DataFrame, flow_col: str = 'Qout', **model_timeseries: pd.DataFrame):
    """Calculate error metrics for n number of model time series compared to gauge data. Takes named dataframes
       as keyword arguments and uses the names given to identify stats for the dataframes in the index of the output df

    Args:
        gauge_data (pd.DataFrame): The gauge data time series.
        flow_col (str, optional): The name of the column representing the flow. Default is 'Qout'.
        **model_timeseries (pd.DataFrame): Keyword arguments representing model time series.

    Returns:
        pd.DataFrame: A DataFrame containing calculated error metrics for each model time series.
    """
    # Add a column for each stat
    metrics_df = pd.DataFrame(columns=STATS_LIST)

    # Add row for each provided model timeseries
    for model_name, model_df in model_timeseries.items():
        stats_dict = {}
        # Calculate each stat using HydroErr and store in dictionary, then add as a row to the output dataframe
        for stat in STATS_LIST:
            stat_calc = getattr(Hs, stat)
            stat_val = stat_calc(model_df[flow_col], gauge_data[flow_col])
            stats_dict[stat] = stat_val
        stats_df = pd.DataFrame(stats_dict, index=[model_name])
        metrics_df = pd.concat([metrics_df, stats_df])

    # Rename columns based on the longer names defined in STATS_NAMES
    return metrics_df.rename(columns={short: long for short, long in zip(STATS_LIST, STATS_NAMES)})


def find_file_recursively(folder_path, target_filename):
    """Recursively search for a file with a specific name in a folder and its subdirectories.

    Args:
        folder_path (str): The path of the folder to start the search.
        target_filename (str): The name of the file to search for.

    Returns:
        str or None: The full path of the found file, or None if not found.
    """
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file == target_filename:
                return os.path.join(root, file)
    return None

def get_vpu_nc(vpu_dir):
    """Open and return a NetCDF dataset corresponding to a region or vpu from the specified directory.

    Args:
        vpu_dir (str): Path to the directory containing NetCDF files.

    Returns:
        xr.Dataset: A NetCDF dataset.
    """
    return xr.open_mfdataset(os.path.join(vpu_dir, 'Qout*.nc'))


def calc_metrics_on_all_gauges(assigned_gauges: pd.DataFrame):
    """Calculate metrics on all gauges based on assigned gauge data.

    Args:
        assigned_gauges (pd.DataFrame): DataFrame containing assigned gauge data.

    Returns:
        pd.DataFrame: DataFrame containing calculated metrics for all gauges and models.
    """
    # Prepare gauge assignments dataframe and lookup mechanisms to match ids to region and vpu codes
    gid_col = 'gauge_id'
    v1_id_col = 'geoglows_id'
    v2_id_col = 'nga_id'
    v2_lookup = pd.read_parquet(V2_LOOKUP_TBL)
    assigned_gauges['v1_region'] = assigned_gauges[v1_id_col].apply(reach_to_region)
    assigned_gauges['v2_vpu_code'] = assigned_gauges[v2_id_col].apply(
        lambda x: v2_lookup.loc[v2_lookup['TDXHydroLinkNo'] == x, 'VPUCode'].values[0])

    # Sort by v1 region, then v2 vpu code, so that already opened timeseries can be reused if necessary for gauges in
    # the same region
    assigned_gauges.sort_values(by=['v1_region', 'v2_vpu_code'], inplace=True)

    # Make a copy in which to write the stats columns
    out_df = assigned_gauges.copy()
    old_v1_region, old_v2_vpu_code = ('', '')

    # Calculate error metrics for each gauge
    for i, gauge_row in assigned_gauges.iterrows():
        gauge_id = gauge_row[gid_col].split('_')[-1]
        v1_id = int(gauge_row[v1_id_col])
        v2_id = int(gauge_row[v2_id_col])
        v1_region = gauge_row['v1_region']
        v2_vpu_code = gauge_row['v2_vpu_code']

        # Only read the historical simulation data if the gauge is in a different region/vpu than the last one
        if v1_region != old_v1_region:
            if not os.path.exists(os.path.join(V1_DIR, v1_region)):
                print(f'Model data for V1 region {v1_region} not found')
                continue
            print(v1_region)
            v1_hist_sim = get_vpu_nc(os.path.join(V1_DIR, v1_region)).sel(rivid=v1_id, nv=0) \
                .to_dataframe().reset_index()
        if v2_vpu_code != old_v2_vpu_code:
            if not os.path.exists(os.path.join(V2_DIR, str(v2_vpu_code))):
                print(f'Model data for V2 vpu {v2_vpu_code} not found')
                continue
            v2_hist_sim = get_vpu_nc(os.path.join(V2_DIR, str(v2_vpu_code))).sel(rivid=v2_id, nv=0) \
                .to_dataframe().reset_index()
        old_v1_region = v1_region
        old_v2_vpu_code = v2_vpu_code

        # Find file with the gauge_id (may need to improve to use the source name if a code is found to not be unique)
        gauge_path = find_file_recursively(GAUGE_DIR, f'{gauge_id}.csv')
        if not gauge_path:
            print(f'Observed data for gauge {gauge_row[gid_col]} not found')
            continue
        gauge_timeseries = pd.read_csv(gauge_path)
        if 'streamflow' not in gauge_timeseries.columns[1].lower():
            print(f'Data for gauge {gauge_row[gid_col]} not streamflow. Skipping...')
            continue

        # Preprocess data for error calculation
        gauge_timeseries = gauge_timeseries.rename(columns={gauge_timeseries.columns[1]: "Qout"})
        gauge_timeseries[gauge_timeseries.columns[0]] = pd.to_datetime(gauge_timeseries[gauge_timeseries.columns[0]])

        # Pare gauge data down to what both historical sims have, then fil
        for sim in [v1_hist_sim, v2_hist_sim]:
            sim.rename(columns={'time': 'sim_time'}, inplace=True)
            gauge_timeseries = gauge_timeseries.merge(
                sim['sim_time'], left_on=gauge_timeseries.columns[0], right_on='sim_time', how='inner') \
                .reset_index(drop=True).drop('sim_time', axis=1)
        v1_hist_sim = v1_hist_sim.loc[v1_hist_sim['sim_time'].isin(gauge_timeseries.iloc[:, 0])]
        v2_hist_sim = v2_hist_sim.loc[v2_hist_sim['sim_time'].isin(gauge_timeseries.iloc[:, 0])]

        # Get metrics dataframe
        metrics = calc_error_metrics(gauge_timeseries, geoglows_v1=v1_hist_sim, geoglows_v2=v2_hist_sim)

        # Reset the index and reshape the DataFrame to be one row
        metrics = metrics.stack().to_frame().T

        # Combine index and column names to get a column for each stat and for each model run
        metrics.columns = [f'{index} - {column}' for index, column in metrics.columns]

        if not metrics.columns.isin(out_df.columns).all():
            out_df[metrics.columns] = -1
        out_df.loc[out_df.index[i], metrics.columns] = metrics.values

    return out_df

import os
from glob import glob
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

from geoglows.streamflow import reach_to_region
import HydroErr.HydroErr as Hs

V1_DIR = 'geoglows1/outputs/'
V2_DIR = 'geoglows2/outputs/'
V2_LOOKUP_TBL = 'master_table.parquet'
STATS_LIST = ['me', 'rmse', 'kge', 'r_squared', 'nse', 'mape']
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
    metrics_df = pd.DataFrame(columns=STATS_LIST)
    for model_name, model_df in model_timeseries.items():
        stats_dict = {}
        for stat in STATS_LIST:
            stat_calc = getattr(Hs, stat)
            stat_val = stat_calc(model_df[flow_col], gauge_data[flow_col])
            stats_dict[stat] = stat_val
        stats_df = pd.DataFrame(stats_dict, index=[model_name])
        metrics_df = pd.concat([metrics_df, stats_df])

    return metrics_df


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
    gid_col = 'gauge_id'
    v1_id_col = 'geoglows_id'
    v2_id_col = 'nga_id'
    v2_lookup = pd.read_parquet(V2_LOOKUP_TBL)
    assigned_gauges['v1_region'] = assigned_gauges[v1_id_col].apply(reach_to_region)
    assigned_gauges['v2_vpu_code'] = assigned_gauges[v2_id_col].apply(
        lambda x: v2_lookup.loc[v2_lookup['TDXHydroLinkNo'] == x, 'VPUCode'].values[0])
    assigned_gauges.sort_values(by=['v1_region', 'v2_vpu_code'], inplace=True)
    out_df = assigned_gauges.copy()
    old_v1_region, old_v2_vpu_code = ''

    for i, gauge_row in assigned_gauges.iterrows():
        gauge_id = gauge_row[gid_col]
        v1_id = int(gauge_row[v1_id_col])
        v2_id = int(gauge_row[v2_id_col])
        v1_region = gauge_row['v1_region']
        v2_vpu_code = gauge_row['v2_vpu_code']

        if v1_region != old_v1_region:
            v1_hist_sim = get_vpu_nc(os.path.join(V1_DIR, v1_region)).sel(rivid=v1_id, nv=0) \
                .to_dataframe().reset_index()
        if v2_vpu_code != old_v2_vpu_code:
            v2_hist_sim = get_vpu_nc(os.path.join(V2_DIR, v2_vpu_code)).sel(rivid=v2_id, nv=0) \
                .to_dataframe().reset_index()
        old_v1_region = v1_region
        old_v2_vpu_code = v2_vpu_code

        gauge_timeseries = pd.read_csv(f'{gauge_id}.csv')

        metrics = calc_error_metrics(gauge_timeseries, geoglows_v1=v1_hist_sim, geoglows_v2=v2_hist_sim)

        # Reset the index and reshape the DataFrame
        metrics = metrics.stack().to_frame().T

        # Combine index and column names
        metrics.columns = [f'{index}_{column}' for index, column in metrics.columns]

        out_df.loc[gauge_row.index, metrics.columns] = metrics

    return out_df
import os
from glob import glob

import geopandas as gpd
import pandas as pd

from matchPoints import match_new_to_old_rivers, match_rivers_to_gauges

pd.set_option('display.max_columns', 20)
gauge_table = pd.read_csv('gauge_table_filters.csv')
total_rows = int(5.6e6)
chunk_size = int(5e5)
region_map = {
    'africa': '100',
    'europe': '200',
    'central_asia': '400',
    'east_asia': '400',
    'west_asia': '400',
    'middle_east': '400',
    'japan': '400',
    'south_asia': '400',
    'australia': '500',
    'islands': '500',
    'south_america': '600',
    'north_america': '700',
    'central_america': '700',
}

out_tbl = pd.DataFrame()

# for geoglows in glob('geoglows/*/*.shp'):
#     region = geoglows.split('/')[-2].split('-geoglows-')[0]
#     nga_code = region_map[region]
#     df = gpd.read_file(geoglows)
#     nga_files = glob(f'tdxhydro/vpu_{nga_code}/*.gpkg')
#     print(f'Assigning streams from file: {geoglows} of length: {df.shape[0]}')
#     for file in nga_files:
#         nga = gpd.read_file(file)
#         print(f'Assigning streams from file: {file} of length: {nga.shape[0]}')
#         if out_tbl.empty:
#             out_tbl = match_new_to_old_rivers(df, nga, gauge_table, new_strmid='TDXHydroLinkNo')
#         else:
#             assigned = match_new_to_old_rivers(df, nga, gauge_table, new_strmid='TDXHydroLinkNo')
#             if assigned is not None:
#                 out_tbl = pd.concat([out_tbl, assigned],
#                                     axis=0).reset_index(drop=True)
#             print(out_tbl)
#         out_tbl.to_csv('gauge_assignments/whole_world_gauge_assign.csv')
#     del df

nga_files = glob(f'tdxhydro/vpu_*/*.gpkg')
for file in nga_files:
    nga = gpd.read_file(file)
    print(f'Assigning streams from file: {file} of length: {nga.shape[0]}')
    if out_tbl.empty:
        out_tbl = match_rivers_to_gauges(nga, gauge_table, new_strmid='TDXHydroLinkNo')
    else:
        assigned = match_rivers_to_gauges(nga, gauge_table, new_strmid='TDXHydroLinkNo')
        if assigned is not None:
            out_tbl = pd.concat([out_tbl, assigned],
                                axis=0).reset_index(drop=True)
        print(out_tbl)
    out_tbl.to_csv('gauge_assignments/whole_world_gauge_assign.csv')


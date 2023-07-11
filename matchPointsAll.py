import os
from glob import glob
import pandas as pd
import geopandas as gpd

from matchPoints import match_new_to_old_rivers

gauge_table = pd.read_csv('gauge_table_filters.csv')
nga = gpd.read_parquet('global_streams_simplified.geoparquet')

for geoglows in glob('geoglows/*/*.shp'):
    df = gpd.read_file(geoglows)
    out_tbl = match_new_to_old_rivers(df, nga, gauge_table)